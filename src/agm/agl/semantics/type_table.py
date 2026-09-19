"""Shared nominal type-declaration table for AgL.

``RecordType``/``EnumType``/``ExceptionType`` (see ``semantics.types``) are
lightweight handles — identified by the declaration they name (``decl_id``),
with ``(module_id, scope_path, name, type_args)`` for records/enums and
``(module_id, scope_path, name)`` for exceptions (never generic) as
resolution/display metadata — carrying no field/variant data of their own.
This module holds the single source of truth for their shapes: a table of
``TypeDef`` templates keyed by declaration identity (``DeclId``), populated by
the type builder as each declaration is resolved. A separate name index maps
each ``(module_id, scope_path, name)`` path (``DeclKey``) to the identity of
the newest declaration registered under it, so two declarations of the same
name path can coexist in the table — a name is a pointer to the newest
declaration bearing it, and an existing reference to a superseded declaration
keeps resolving to that declaration's own shape.

``TypeDef`` stores field/variant type *templates*: finite ``Type`` trees that
may reference the declaration's own type parameters via ``TypeVarType`` nodes
— the same kind of template already computed for generic types today
(``typecheck.env.GenericTypeDef.template``), just captured under one
representation shared by records, enums, and exceptions.
``TypeTable.record_fields``/``enum_members`` substitute a handle's
``type_args`` into those templates and memoize the result per handle;
``TypeTable.exception_fields`` has no ``type_args`` to substitute but instead
flattens the ``extends`` base chain into one field mapping. The table also
keeps plain ``MethodDef`` data keyed by nominal owner identity or by a built-in
receiver constructor. Method candidates retain every declaration sharing an
owner and name, and expose exception inheritance as nearest-first levels.

``comparable_types``/``_reaches_non_data`` live here rather than in
``semantics.types`` because their record/enum/exception arms consult the
table's declaration-level non-data-reachability flags instead of walking
embedded fields; ``semantics.types`` cannot import this module without a
circular import. The flags themselves are a fixpoint over the whole table
(``semantics.analyses.compute_non_data_reachability``, cycle-safe by
construction), cached on :class:`TypeTable` and invalidated whenever the
table's declarations change. That one fact answers two separate language
questions — may ``=``/``!=`` be applied (``comparable_types``)? and is there
a JSON representation (:meth:`TypeTable.nominal_is_json_convertible`)? —
which is why it is named for the fact rather than for either consumer.

:meth:`TypeTable.has_finite_schema` answers a related but distinct
whole-type question: not "does this type
support ``=``?" but "is this type's reachable *instantiation closure* finite
(so it has a finite JSON schema)?" — a generic recursive declaration may
reference itself at ever-larger arguments (polymorphic recursion), which
never blocks construction/matching/equality but does mean no finite schema
exists. Backed by ``semantics.analyses.compute_finite_closure``, cached and
invalidated the same way as the non-data-reachability fixpoint.
:meth:`TypeTable.first_infinite_declaration`/:meth:`TypeTable.no_finite_schema_message`
build on the same query to name the culprit declaration for a use-site
diagnostic (agent output target, cast target, parameter type).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, assert_never, cast

from agm.agl.ir.reserved_nominals import (
    NO_DECL_ID,
    require_reserved_enum_member_id,
)
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id as _reserved_id
from agm.agl.modules.ids import RESERVED_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.external_names import NO_EXTERNAL_NAME, ExternalName
from agm.agl.semantics.types import (
    HOST_MINTED_PRELUDE_TYPE_IDS,
    HOST_MINTED_PRELUDE_TYPE_NAMES,
    ArrayType,
    BoolType,
    BottomType,
    CastKind,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeTemplate,
    TypeTemplateMatch,
    TypeVarType,
    UnitType,
    contains_type_var,
    free_type_vars,
    is_assignable,
    is_scalar_json_shaped,
    match_nominal_owner_template,
    spells_bare,
    standard_option_type,
    substitute,
    type_children,
)
from agm.agl.zones import ParamZone
from agm.util.graph import bfs_first

if TYPE_CHECKING:
    from agm.agl.semantics.analyses import FiniteClosure, NonDataReachability

TypeDefKind = Literal["record", "enum", "exception"]
#: A declaration's name path — ``(module_id, scope_path, name)``. Used only
#: for name resolution (the ``TypeTable`` name index, ``get``); it is not a
#: declaration's identity (see :data:`DeclId`), since more than one
#: declaration may share a name path over a table's lifetime.
DeclKey = tuple[ModuleId, tuple[str, ...], str]
#: A declaration's identity: an AST node id, a reserved id (see
#: ``ir.reserved_nominals``), or ``NO_DECL_ID`` when none is attached. This is
#: the ``TypeTable``'s real key — see the module docstring.
DeclId = int
NominalOwner = RecordType | EnumType | ExceptionType


@dataclass(frozen=True, slots=True)
class MethodDef:
    """Plain declaration data for one nominal or built-in receiver method.

    ``module_id``/``scope_path``/``name`` and ``decl_node_id`` identify the
    declared function, not its owner: a root ``Point`` method ``Point::move``
    has declaration scope ``("Point",)``, while the table files it under the
    ``Point`` DECLARATION's own identity (see :meth:`TypeTable.register_method`).
    ``signature`` is the method's ordinary function type, including its
    receiver as the first parameter. ``receiver_type_param_arity`` records how
    many leading method type parameters belong to the receiver type.
    """

    module_id: ModuleId
    scope_path: tuple[str, ...]
    name: str
    decl_node_id: int
    signature: FunctionType
    receiver_type_param_arity: int
    type_params: tuple[str, ...] = ()
    is_builtin: bool = False

    @property
    def declaration_key(self) -> DeclKey:
        """Return the method's structured declaration identity."""
        return self.module_id, self.scope_path, self.name


@dataclass(frozen=True, slots=True)
class TypeDef:
    """One nominal type declaration's parameter list and shape templates.

    ``fields``/``members`` are stored as tuples so ``TypeDef`` stays hashable
    and declaration order is explicit. ``members`` contains record handles;
    the record declarations own their fields. ``TypeTable`` substitutes a
    handle's ``type_args`` into these templates and caches the result.

    ``fields``  — field templates for records; for exceptions, the
                  exception's OWN field templates only — NOT flattened with
                  the base chain (see :meth:`TypeTable.exception_fields`).
    ``members`` — record type templates for enums (empty for
                  records/exceptions).
    ``mutable_fields`` — names of the ``var`` fields a record declares (a
                  subset of ``fields``); always empty for enums and
                  exceptions, neither of which admits a mutable field.
    ``abstract`` — exception metadata: ``True`` for the hierarchy root
                   (catchable but not constructible); unused for
                   records/enums.
    ``base``     — exception metadata: the resolved declaration identity
                   (``DeclId``) of the ``extends`` target, or ``None`` for the
                   root; unused for records/enums.
    ``field_kinds`` — each field's own ``ParamZone``, in ``fields`` order;
                   for an exception, its OWN kinds only (see
                   :meth:`TypeTable.field_kinds` for the flattened base
                   chain). Mirrors the ``TypeEnvironment`` constructor-kind
                   registry, computed once at the same declaration site and
                   fed to both.
    ``is_builtin`` — ``True`` when this entry came from a source ``builtin``
                   declaration, at whatever path it was written. It is
                   metadata about the declaration, not part of its shape, so
                   it is excluded from equality/hashing (``compare=False``):
                   :meth:`_TypeBuilder._validate_builtin_shape` compares a
                   ``builtin`` declaration's whole ``TypeDef`` against a
                   seeded canonical literal, which never sets this flag.
    ``decl_node_id`` — the identity of the declaration this ``TypeDef``
                   describes (an AST node id, or a reserved id for a
                   host-known built-in name — see ``ir.reserved_nominals``),
                   or ``NO_DECL_ID`` when none is attached. Like
                   ``is_builtin``, it is metadata about the declaration
                   rather than part of its shape, so it too is excluded from
                   equality/hashing (``compare=False``) for the same reason:
                   :meth:`_TypeBuilder._validate_builtin_shape`'s comparison
                   against a seeded canonical literal must not fail merely
                   because the two carry different declaration identities.
    ``is_inline_enum_member`` — ``True`` for a synthetic record declaration
                   created by an inline enum member. Its inhabitation is
                   determined by its enclosing enum rather than independently.
    ``external_name`` — a record's own ``@name``/``@json-name`` spellings
                   (its value-syntax name and its ``$case`` tag as an enum
                   member); unused for enums and exceptions.
    ``field_external_names`` — ``(field_name, ExternalName)`` pairs for the
                   OWN fields carrying ``@name``/``@json-name``, in
                   declaration order; unrenamed fields are absent.
    ``doc``        — the declaration's recognized ``@doc`` prose, when present.
                   It is presentation metadata rather than part of the type's
                   semantic shape, so it is excluded from equality/hashing.
    """

    kind: TypeDefKind
    name: str
    module_id: ModuleId
    scope_path: tuple[str, ...] = ()
    type_params: tuple[str, ...] = ()
    fields: tuple[tuple[str, Type], ...] = ()
    mutable_fields: frozenset[str] = frozenset()
    members: tuple[RecordType, ...] = ()
    abstract: bool = False
    base: DeclId | None = None
    field_kinds: tuple[ParamZone, ...] = ()
    is_builtin: bool = field(default=False, compare=False)
    decl_node_id: int = field(default=NO_DECL_ID, compare=False)
    is_inline_enum_member: bool = field(default=False, compare=False)
    external_name: ExternalName = NO_EXTERNAL_NAME
    field_external_names: tuple[tuple[str, ExternalName], ...] = ()
    doc: str | None = field(default=None, compare=False)

    def handle(self, type_args: tuple[Type, ...] = ()) -> RecordType | EnumType | ExceptionType:
        """Return the ``RecordType``/``EnumType``/``ExceptionType`` handle naming this ``TypeDef``.

        Convenience for call sites that hold a ``TypeDef`` and need the
        corresponding handle (e.g. to register a value, or to pass to
        :meth:`TypeTable.record_fields`/:meth:`TypeTable.enum_members`/
        :meth:`TypeTable.exception_fields`). *type_args* defaults to ``()``
        for non-generic defs and must be empty for an exception (exceptions
        are never generic) — passing a non-empty tuple for one raises
        ``ValueError``. The returned handle's ``decl_id`` is stamped from
        ``self.decl_node_id``.
        """
        match self.kind:
            case "record":
                return RecordType(
                    name=self.name,
                    type_args=type_args,
                    module_id=self.module_id,
                    scope_path=self.scope_path,
                    decl_id=self.decl_node_id,
                )
            case "enum":
                return EnumType(
                    name=self.name,
                    type_args=type_args,
                    module_id=self.module_id,
                    scope_path=self.scope_path,
                    decl_id=self.decl_node_id,
                )
            case "exception":
                if type_args:
                    raise ValueError("TypeDef.handle() does not accept type_args for an exception")
                return ExceptionType(
                    name=self.name,
                    module_id=self.module_id,
                    scope_path=self.scope_path,
                    decl_id=self.decl_node_id,
                )
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)


class TypeTable:
    """Mutable registry of ``TypeDef``s keyed by declaration identity (``DeclId``).

    Populated by the type builder as each declaration's body is resolved; a
    single instance is shared across a module graph's per-module environments
    so every module's declarations land in the same table. A name index maps
    each ``(module_id, scope_path, name)`` path to the identity of the newest
    declaration registered under it (see :meth:`register`), so a name lookup
    (:meth:`get`) always answers with the newest declaration while an older
    declaration remains retrievable by identity (:meth:`get_by_id`) — a
    superseded declaration is retained, never removed.
    """

    def __init__(self) -> None:
        self._defs: dict[DeclId, TypeDef] = {}
        # Name path -> the identity of the newest declaration registered
        # under it. A name lookup (get) always resolves through this index.
        self._name_index: dict[DeclKey, DeclId] = {}
        # Identities of declarations that never took effect (see :meth:`orphan`).
        # Retained in ``_defs`` so their linked descriptors stay derivable, but
        # excluded from every query about what the session actually declares.
        self._orphaned: set[DeclId] = set()
        self._record_fields_cache: dict[DeclId, dict[RecordType, Mapping[str, Type]]] = {}
        self._enum_members_cache: dict[DeclId, dict[EnumType, tuple[RecordType, ...]]] = {}
        self._enum_member_names_cache: dict[DeclId, dict[EnumType, Mapping[str, RecordType]]] = {}
        self._enum_member_by_decl_cache: dict[
            DeclId, dict[EnumType, Mapping[DeclId, RecordType]]
        ] = {}
        # Exceptions are non-generic, so (unlike record_fields/enum_members)
        # there is no type_args substitution — the memo is keyed directly by
        # declaration identity, one entry per exception.
        self._exception_fields_cache: dict[DeclId, Mapping[str, Type]] = {}
        # Memo for field_kinds's exception branch — same keying convention as
        # _exception_fields_cache above.
        self._exception_field_kinds_cache: dict[DeclId, tuple[tuple[str, ParamZone], ...]] = {}
        # Whole-table indexes over the live standard-library builtin declarations.
        # Both answer questions about what the session declares as a whole, so
        # they are invalidated wholesale like the fixpoints below.
        self._standard_builtins: dict[str, TypeDef] | None = None
        self._host_minted_ids: frozenset[DeclId] | None = None
        # Methods are independent plain declaration data, keyed by their
        # nominal owner's identity rather than by an import environment. Each
        # name preserves every declaration identity that contributes it.
        self._methods: dict[DeclId, dict[str, dict[DeclKey, MethodDef]]] = {}
        self._builtin_methods: dict[str, dict[str, dict[DeclKey, MethodDef]]] = {}
        # Whole-table non-data-reachability fixpoint (see
        # :meth:`nominal_reaches_non_data`), computed lazily on first use and
        # invalidated (set back to ``None``) whenever a declaration is added,
        # removed, or overwritten.
        self._non_data_caps: NonDataReachability | None = None
        # Member declaration id -> the enums declaring or referencing it, in
        # registration order. A referenced member belongs to several enums, so
        # this is multi-valued. Whole-table, rebuilt on any registration change.
        self._member_enum_owners: dict[DeclId, tuple[DeclId, ...]] | None = None
        # Whole-table finiteness fixpoint (see :meth:`has_finite_schema`),
        # cached and invalidated the same way as ``_non_data_caps``.
        self._finite_closure: FiniteClosure | None = None

    def register(self, typedef: TypeDef) -> None:
        """Register *typedef* under its own declaration identity.

        A registered declaration always carries an identity: when
        self-validation is enabled, a *typedef* whose ``decl_node_id`` is
        ``NO_DECL_ID`` is rejected outright — an unregistered identity would
        otherwise silently pass ``TypeTable.register`` and leave every later
        identity-keyed lookup for it (including a fresh, unrelated caller
        that also forgot to stamp one) irretrievable.

        Registering a *different* definition under an already-registered
        identity is an internal invariant violation — every declaration is
        built exactly once, so, when self-validation is enabled, this raises
        ``AssertionError`` rather than a user-facing diagnostic. Re-checking
        the identical declaration again (e.g. the REPL re-checking a promoted
        entry against a fresh environment, or the program pre-pass and the
        per-module check both building the same module) is expected and is
        silently accepted. With self-validation disabled (the production
        path), a re-registration under an existing identity is always
        silently accepted, matching declaration reuse rather than
        re-verifying it.

        Registering a new identity under a name path that another identity
        already occupies does not remove the older declaration — it stays
        registered, and retrievable by its own identity — but repoints the
        name index (:meth:`get`) at the registered one: a name is a pointer
        to the declaration that most recently claimed it, not the
        declaration itself. Registering an already-registered identity
        reclaims its name path the same way, which is how a caller restores
        a name to a declaration that a since-discarded one took over.
        """
        if self_validation_enabled() and typedef.decl_node_id == NO_DECL_ID:
            raise AssertionError(
                f"cannot register a TypeDef with no declaration identity: {typedef!r}"
            )
        decl_id = typedef.decl_node_id
        existing = self._defs.get(decl_id)
        self._name_index[(typedef.module_id, typedef.scope_path, typedef.name)] = decl_id
        if existing is None:
            self._defs[decl_id] = typedef
            self._standard_builtins = None
            self._host_minted_ids = None
            self._non_data_caps = None
            self._member_enum_owners = None
            self._finite_closure = None
            return
        if self_validation_enabled() and existing != typedef:
            raise AssertionError(
                f"conflicting TypeDef registration for identity {decl_id!r}: "
                f"{existing!r} is already registered, got {typedef!r}"
            )

    def get(
        self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()
    ) -> TypeDef | None:
        """Return the newest registered ``TypeDef`` for a name path, via the name index."""
        decl_id = self._name_index.get((module_id, scope_path, name))
        return None if decl_id is None else self._defs.get(decl_id)

    def get_by_id(self, decl_id: DeclId) -> TypeDef | None:
        """Return the registered ``TypeDef`` for *decl_id*, or ``None`` if unregistered.

        Unlike :meth:`get`, this reaches a superseded declaration directly by
        its own identity, independent of whichever declaration the name index
        currently points a shared name path at.
        """
        return self._defs.get(decl_id)

    def orphan(self, decl_id: DeclId) -> None:
        """Record that *decl_id*'s declaration never took effect, and release its name.

        For a declaration an incremental entry failed before promoting (see
        :meth:`~agm.agl.typecheck.env.TypeEnvironment.restore_type_names_from`).
        No value of one can exist — a later item of the same entry could not
        have promoted either — so nothing may resolve to it and no
        whole-table query about what the session declares may answer with it
        (:meth:`builtin_declaration`, and the exception descendant scan behind
        member-conflict diagnostics).

        This is emphatically NOT what a redeclaration does: a SUPERSEDED
        declaration stays live here, because values built from it survive and
        its own members still constrain what may be declared around it. Only
        a caller that knows a declaration never took effect may orphan it.

        The declaration itself stays registered under its own identity. The
        link image's nominal descriptors are DERIVED from this table on every
        lowering (``lower.program``), so a declaration that vanished outright
        would strand the descriptor the failed entry already linked, leaving
        it claiming a name path it no longer holds. Releasing the name
        instead lets the next lowering correct that descriptor.

        *decl_id* must be registered.
        """
        typedef = self._defs[decl_id]
        self._orphaned.add(decl_id)
        key = (typedef.module_id, typedef.scope_path, typedef.name)
        if self._name_index.get(key) == decl_id:
            del self._name_index[key]
        self._invalidate_cache_for(decl_id)

    def is_current(self, typedef: TypeDef) -> bool:
        """Return whether *typedef*'s identity is the one its name path currently resolves to.

        Answers "does this identity currently bear its name path?" via the
        name index (see the class and module docstrings) rather than via
        insertion order: a superseded declaration and a declaration from an
        unpromoted REPL entry both remain registered under their own
        identity (:meth:`get_by_id` still reaches them), but neither one is
        what :meth:`get` resolves their shared name path to any more, so
        this returns ``False`` for both.
        """
        key = (typedef.module_id, typedef.scope_path, typedef.name)
        return self._name_index.get(key) == typedef.decl_node_id

    @staticmethod
    def _method_sort_key(method: MethodDef) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        return method.module_id.segments, method.scope_path, method.name

    @classmethod
    def _method_level(cls, methods: Mapping[DeclKey, MethodDef]) -> tuple[MethodDef, ...]:
        return tuple(sorted(methods.values(), key=cls._method_sort_key))

    def _remove_method_declaration(self, declaration_key: DeclKey) -> None:
        """Remove a superseded method declaration from every receiver."""
        for methods in self._methods.values():
            for candidates in methods.values():
                candidates.pop(declaration_key, None)
        for methods in self._builtin_methods.values():
            for candidates in methods.values():
                candidates.pop(declaration_key, None)

    def _put_method(self, decl_id: DeclId, method: MethodDef) -> None:
        """Register *method* under its declaration key on *decl_id*."""
        self._remove_method_declaration(method.declaration_key)
        self._methods.setdefault(decl_id, {}).setdefault(method.name, {})[
            method.declaration_key
        ] = method

    def register_method(self, owner: NominalOwner, method: MethodDef) -> None:
        """Register *method* under its nominal *owner*."""
        self._put_method(owner.decl_id, method)

    def register_builtin_method(self, constructor: str, method: MethodDef) -> None:
        """Register *method* under a built-in receiver type constructor."""
        self._remove_method_declaration(method.declaration_key)
        self._builtin_methods.setdefault(constructor, {}).setdefault(method.name, {})[
            method.declaration_key
        ] = method

    def restore_methods_from(self, previous: TypeTable, declaration_ids: Collection[int]) -> None:
        """Restore methods replaced by unpromoted declarations from *previous*."""
        declaration_keys = {
            method.declaration_key
            for methods in self._methods.values()
            for candidates in methods.values()
            for method in candidates.values()
            if method.decl_node_id in declaration_ids
        } | {
            method.declaration_key
            for methods in self._builtin_methods.values()
            for candidates in methods.values()
            for method in candidates.values()
            if method.decl_node_id in declaration_ids
        }
        if not declaration_keys:
            return
        for methods in self._methods.values():
            for candidates in methods.values():
                for key in declaration_keys:
                    candidates.pop(key, None)
        for methods in self._builtin_methods.values():
            for candidates in methods.values():
                for key in declaration_keys:
                    candidates.pop(key, None)
        for decl_id, methods in previous._methods.items():
            for candidates in methods.values():
                for key, method in candidates.items():
                    if key in declaration_keys:
                        self._put_method(decl_id, method)
        for constructor, methods in previous._builtin_methods.items():
            for candidates in methods.values():
                for key, method in candidates.items():
                    if key in declaration_keys:
                        self.register_builtin_method(constructor, method)

    @staticmethod
    def _builtin_constructor(owner: Type) -> str | None:
        """Return the method-table key for a structural or scalar built-in type."""
        if isinstance(owner, ArrayType):
            return "array"
        if isinstance(owner, DictType):
            return "dict"
        if isinstance(owner, TextType):
            return "text"
        if isinstance(owner, JsonType):
            return "json"
        if isinstance(owner, IntType):
            return "int"
        if isinstance(owner, DecimalType):
            return "decimal"
        if isinstance(owner, BoolType):
            return "bool"
        return None

    def method_candidates(
        self, owner: NominalOwner | Type, name: str
    ) -> tuple[tuple[MethodDef, ...], ...]:
        """Return same-owner candidates for *name*, nearest exception level first."""
        constructor = self._builtin_constructor(owner)
        if constructor is not None:
            return (self._method_level(self._builtin_methods.get(constructor, {}).get(name, {})),)
        if not isinstance(owner, (RecordType, EnumType, ExceptionType)):
            return ()
        owner_ids: tuple[DeclId, ...] = (owner.decl_id,)
        if isinstance(owner, ExceptionType):
            owner_ids += tuple(base.decl_node_id for base in self.ancestor_defs(owner.decl_id))
        return tuple(
            self._method_level(self._methods.get(decl_id, {}).get(name, {}))
            for decl_id in owner_ids
        )

    def declared_methods(self, owner_id: DeclId) -> Mapping[str, MethodDef]:
        """Return the direct method chosen for each name declared on *owner_id*."""
        return {
            name: self._method_level(methods)[0]
            for name, methods in self._methods.get(owner_id, {}).items()
            if methods
        }

    def _exception_chain(self, decl_id: DeclId, *, caller: str) -> list[tuple[DeclId, TypeDef]]:
        """Return *decl_id*'s base chain, base first, rejecting a cyclic base link.

        Every flattened exception accessor inherits base-first declaration order
        from this one walk, so ``caller`` only selects the ``KeyError``/
        ``AssertionError`` label of the accessor that asked.
        """
        chain: list[tuple[DeclId, TypeDef]] = []
        visited: set[DeclId] = set()
        current: DeclId | None = decl_id
        while current is not None:
            if current in visited:
                raise AssertionError(f"cyclic exception base chain detected at {current!r}")
            visited.add(current)
            typedef = self._require_exception_def(current, caller=caller)
            chain.append((current, typedef))
            current = typedef.base
        chain.reverse()
        return chain

    def ancestor_defs(self, decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return *decl_id*'s exception ancestors, nearest first.

        Empty for a hierarchy root and for any non-exception declaration, so a
        caller checking inherited members needs no base-chain walk of its own.
        """
        typedef = self._defs.get(decl_id)
        assert typedef is not None, f"no TypeDef registered for identity {decl_id!r}"
        if typedef.kind != "exception" or typedef.base is None:
            return ()
        chain = self._exception_chain(typedef.base, caller="ancestor_defs")
        return tuple(base_def for _base_id, base_def in reversed(chain))

    def is_exception_ancestor(self, ancestor_id: DeclId, decl_id: DeclId) -> bool:
        """Whether *ancestor_id* is in *decl_id*'s exception base chain."""
        return any(ancestor.decl_node_id == ancestor_id for ancestor in self.ancestor_defs(decl_id))

    def is_builtin_exception_root(self, decl_id: DeclId) -> bool:
        """Whether *decl_id* is the built-in ``Exception`` root: reserved or ``builtin``, no base.

        Every other built-in exception extends it, so this is its identity in any
        graph, whether the reserved fallback or a loaded standard declaration. A
        user-declared base-less exception is not the root.
        """
        typedef = self._defs.get(decl_id)
        return (
            typedef is not None
            and typedef.kind == "exception"
            and typedef.base is None
            and (typedef.module_id.is_reserved or typedef.is_builtin)
        )

    def _invalidate_cache_for(self, decl_id: DeclId) -> None:
        self._record_fields_cache.pop(decl_id, None)
        self._enum_members_cache.pop(decl_id, None)
        self._enum_member_names_cache.pop(decl_id, None)
        self._enum_member_by_decl_cache.pop(decl_id, None)
        # Exception field accessors flatten inherited base chains, so changing
        # one exception can invalidate cached descendants as well as the changed
        # identity. Clear the exception caches wholesale rather than trying to
        # maintain a reverse-inheritance index.
        self._exception_fields_cache.clear()
        self._exception_field_kinds_cache.clear()
        # The non-data-reachability and finiteness fixpoints are whole-table
        # (any declaration's flag can in principle depend on any other's), so
        # a single changed identity invalidates the whole cached result rather
        # than just this one.
        self._standard_builtins = None
        self._host_minted_ids = None
        self._non_data_caps = None
        self._member_enum_owners = None
        self._finite_closure = None

    def record_fields(self, handle: RecordType) -> Mapping[str, Type]:
        """Return *handle*'s field types with its ``type_args`` substituted in.

        Memoized per handle: ``RecordType`` equality/hash cover ``decl_id``
        and ``type_args``, so the same handle always maps to the same
        substituted mapping object. The memo is bucketed by ``decl_id`` so a single
        identity's invalidation (:meth:`merge_from`) never has to scan entries
        for other identities.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``decl_id`` — every valid handle is expected to have one.
        Raises ``AssertionError`` if the registered def's ``kind`` is not
        ``"record"`` — an internal-invariant violation, since a ``RecordType``
        handle only ever names a record declaration.
        """
        decl_id = handle.decl_id
        bucket = self._record_fields_cache.get(decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._require_record_def(handle, caller="record_fields")
        subst = dict(zip(typedef.type_params, handle.type_args))
        result: Mapping[str, Type] = {
            fname: substitute(ftype, subst) for fname, ftype in typedef.fields
        }
        self._record_fields_cache.setdefault(decl_id, {})[handle] = result
        return result

    def record_mutable_fields(self, handle: RecordType) -> frozenset[str]:
        """Return the names of *handle*'s ``var`` fields.

        Mutability is declared, so it is neither substituted into nor varied
        by a handle's ``type_args``: every handle for one declaration reads
        the same set straight off its ``TypeDef``, and there is nothing per
        handle to memoize (unlike :meth:`record_fields`).

        Raises the same errors as :meth:`record_fields` for an unregistered
        or non-record handle.
        """
        return self._require_record_def(handle, caller="record_mutable_fields").mutable_fields

    def _require_record_def(self, handle: RecordType, *, caller: str) -> TypeDef:
        typedef = self._defs.get(handle.decl_id)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for record {handle!r}")
        if typedef.kind != "record":
            raise AssertionError(
                f"{caller} called for {handle!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'record'"
            )
        return typedef

    def enum_members(self, handle: EnumType) -> tuple[RecordType, ...]:
        """Return *handle*'s member record types with ``type_args`` substituted in.

        Members retain their declaration identities and field ownership. The
        result is memoized per enum instantiation. Raises ``KeyError`` if no
        definition is registered, or ``AssertionError`` when the identity is
        not an enum.
        """
        decl_id = handle.decl_id
        bucket = self._enum_members_cache.get(decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._defs.get(decl_id)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for enum {handle!r}")
        if typedef.kind != "enum":
            raise AssertionError(
                f"enum_members called for {handle!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'enum'"
            )
        subst = dict(zip(typedef.type_params, handle.type_args))
        result = tuple(cast(RecordType, substitute(member, subst)) for member in typedef.members)
        self._enum_members_cache.setdefault(decl_id, {})[handle] = result
        return result

    def _member_enum_owner_index(self) -> Mapping[DeclId, tuple[DeclId, ...]]:
        """Return the memoized member-declaration -> owning-enum index.

        A member declared in one enum may also be referenced by others, so the
        relation is multi-valued and keeps registration order — the order the
        owner scan used before this index existed.
        """
        index = self._member_enum_owners
        if index is None:
            index = {}
            for enum_def in self._defs.values():
                if enum_def.kind != "enum":
                    continue
                for member in enum_def.members:
                    index[member.decl_id] = (
                        *index.get(member.decl_id, ()),
                        enum_def.decl_node_id,
                    )
            self._member_enum_owners = index
        return index

    def _enum_defs_owning(self, record: RecordType) -> tuple[TypeDef, ...]:
        """Return the enum definitions *record* is a member of, in registration order."""
        owners = self._member_enum_owner_index().get(record.decl_id, ())
        return tuple(self._defs[owner_id] for owner_id in owners)

    @staticmethod
    def _match_enum_member_template(
        enum_def: TypeDef, member: RecordType, record: RecordType
    ) -> TypeTemplateMatch | None:
        """Match a concrete record against one enum member's complete template.

        An enum may have phantom parameters which no member record can reveal,
        so their missing bindings are permitted here. Callers that need a
        concrete enum handle still require bindings for every enum parameter.
        """
        if member.decl_id != record.decl_id:
            return None
        return match_nominal_owner_template(TypeTemplate(member, enum_def.type_params), record)

    def records_share_enum_membership(self, records: tuple[RecordType, ...]) -> bool:
        """Return whether *records* belong to one currently nameable enum instantiation.

        A member record captures only the enum parameters its fields use. The
        records therefore share an enum instantiation when they belong to the
        same current enum and their captured arguments give every shared
        parameter the same value; parameters none of them captures may take
        any value. Superseded enums remain available by identity for retained
        values, but their reused name cannot annotate a new expression.
        """
        candidates = (
            self._enum_defs_owning(records[0])
            if records
            else tuple(enum_def for enum_def in self._defs.values() if enum_def.kind == "enum")
        )
        for enum_def in candidates:
            if not self.is_current(enum_def):
                continue
            members = {member.decl_id: member for member in enum_def.members}
            bindings: dict[str, Type] = {}
            for record in records:
                member = members.get(record.decl_id)
                if member is None:
                    break
                match = self._match_enum_member_template(enum_def, member, record)
                if match is None:
                    break
                for parameter, argument in match.bindings:
                    previous = bindings.setdefault(parameter, argument)
                    if previous != argument:
                        break
                else:
                    continue
                break
            else:
                return True
        return False

    def enum_owners_for_member(self, handle: RecordType) -> tuple[EnumType, ...]:
        """Return every concrete enum containing *handle* with known arguments.

        Referenced records may belong to several enums.  The result therefore
        preserves the full relation instead of making registration order part
        of semantic validation.
        """
        owners: list[EnumType] = []
        for typedef in self._enum_defs_owning(handle):
            # The owner index only yields enums that declare or reference this
            # member, so the lookup always succeeds.
            member = next(item for item in typedef.members if item.decl_id == handle.decl_id)
            match = self._match_enum_member_template(typedef, member, handle)
            if match is None:
                continue
            bindings = dict(match.bindings)
            if len(bindings) != len(typedef.type_params):
                continue
            args = tuple(bindings[param] for param in typedef.type_params)
            result = typedef.handle(args)
            assert isinstance(result, EnumType)
            owners.append(result)
        return tuple(owners)

    def record_matches_enum_member(
        self, enum: EnumType, member_name: str, record: RecordType
    ) -> bool:
        """Return whether *record* matches the named member of *enum* exactly.

        A generic enum's bare template may qualify any of its instantiations,
        so free variables still present in its member template are inferred
        while concrete owner arguments must match exactly.
        """
        member = self.enum_member_names(enum).get(member_name)
        if member is None or member.decl_id != record.decl_id:
            return False
        parameters = tuple(sorted(free_type_vars(member)))
        return match_nominal_owner_template(TypeTemplate(member, parameters), record) is not None

    def is_enum_member(self, handle: RecordType) -> bool:
        """Return whether *handle* names a declaration registered as an enum member.

        This is a declaration-membership query: it remains true for a
        fieldless generic member whose record handle cannot reconstruct its
        owning enum's phantom arguments.
        """
        return handle.decl_id in self._member_enum_owner_index()

    def enum_member_names(self, handle: EnumType) -> Mapping[str, RecordType]:
        """Return the terminal member-name index for one enum instantiation."""
        decl_id = handle.decl_id
        bucket = self._enum_member_names_cache.get(decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        result = {member.name: member for member in self.enum_members(handle)}
        self._enum_member_names_cache.setdefault(decl_id, {})[handle] = result
        return result

    def enum_member_by_decl(self, handle: EnumType, decl_id: DeclId) -> RecordType | None:
        """Return *handle*'s member declared as *decl_id*, or ``None`` if it has none.

        The declaration-keyed counterpart of :meth:`enum_member_names`, memoized
        per enum instantiation the same way. Callers holding a constructor's
        declaration identity use this rather than rescanning the member tuple.
        """
        return self.enum_member_ids(handle).get(decl_id)

    def enum_member_ids(self, handle: EnumType) -> Mapping[DeclId, RecordType]:
        """Return *handle*'s members indexed by declaration id, memoized per instantiation.

        Membership tests and member lookups both key off a declaration id, so
        callers use this index rather than rescanning the member tuple.
        """
        enum_decl_id = handle.decl_id
        bucket = self._enum_member_by_decl_cache.get(enum_decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        result = {member.decl_id: member for member in self.enum_members(handle)}
        self._enum_member_by_decl_cache.setdefault(enum_decl_id, {})[handle] = result
        return result

    def shared_enum_members(self, source: EnumType, target: EnumType) -> tuple[RecordType, ...]:
        """Return *source* members that are also exact members of *target*.

        Constructor identity alone is insufficient for generic members: two
        instantiations of the same constructor are shared only when their
        captured type arguments also match. Source declaration order is
        preserved for deterministic lowering.
        """
        target_members = frozenset(self.enum_members(target))
        return tuple(member for member in self.enum_members(source) if member in target_members)

    def enum_is_subset(self, source: EnumType, target: EnumType) -> bool:
        """Return whether every constructor of *source* is a constructor of *target*."""
        return frozenset(self.enum_members(source)) <= frozenset(self.enum_members(target))

    def exception_fields(self, handle: ExceptionType) -> Mapping[str, Type]:
        """Return *handle*'s fully flattened field types (base chain applied).

        Exceptions are non-generic, so unlike :meth:`record_fields`/
        :meth:`enum_members` there is no ``type_args`` substitution — the
        result is memoized directly per ``decl_id``. Base fields come first
        (the root contributes ``message``), followed by the
        exception's own fields, matching declaration order.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``decl_id``. Raises ``AssertionError`` if the registered def's
        ``kind`` is not ``"exception"``, or if the base chain contains a
        cycle — an internal-invariant violation, since the whole-program
        inhabitation pre-pass rejects ``extends`` cycles as uninhabitable
        before this can fire in production; this guard is for internal
        robustness, not a user diagnostic.
        """
        decl_id = handle.decl_id
        cached = self._exception_fields_cache.get(decl_id)
        if cached is not None:
            return cached
        result = self._flatten_exception_fields(decl_id)
        self._exception_fields_cache[decl_id] = result
        return result

    def _flatten_exception_fields(self, decl_id: DeclId) -> Mapping[str, Type]:
        fields: dict[str, Type] = {}
        for _chain_id, typedef in self._exception_chain(decl_id, caller="exception_fields"):
            fields.update(typedef.fields)
        return fields

    def field_kinds(self, handle: RecordType | ExceptionType) -> tuple[tuple[str, ParamZone], ...]:
        """Return *handle*'s ``(field_name, ParamZone)`` pairs, in field order.

        For a record, reads the zones straight off its own ``TypeDef``
        (:attr:`TypeDef.field_kinds`) — declaration-level, like
        :meth:`record_mutable_fields`, so every instantiation of a generic
        record shares the same zones regardless of ``type_args`` and no
        memoization is needed. ``field_kinds`` must have one entry per field,
        in order — a builder that leaves it unset for a non-empty ``fields``
        is a bug, caught by the strict zip below rather than silently
        defaulted.

        For an exception, mirrors :meth:`exception_fields`'s base-chain
        flattening (base fields first, in declaration order, then the
        exception's own), memoized the same way — an exception's OWN fields
        honor their declared ``@arg-*`` attribute exactly like a record's
        fields do; only inheritance is exception-specific.

        Raises ``KeyError``/``AssertionError`` under the same conditions as
        :meth:`record_fields`/:meth:`exception_fields`.
        """
        if isinstance(handle, ExceptionType):
            decl_id = handle.decl_id
            cached = self._exception_field_kinds_cache.get(decl_id)
            if cached is not None:
                return cached
            result = self._flatten_exception_field_kinds(decl_id)
            self._exception_field_kinds_cache[decl_id] = result
            return result
        typedef = self._require_record_def(handle, caller="field_kinds")
        return tuple(
            zip((fname for fname, _ftype in typedef.fields), typedef.field_kinds, strict=True)
        )

    def _flatten_exception_field_kinds(self, decl_id: DeclId) -> tuple[tuple[str, ParamZone], ...]:
        return tuple(
            (fname, kind)
            for _chain_id, typedef in self._exception_chain(decl_id, caller="field_kinds")
            for (fname, _ftype), kind in zip(typedef.fields, typedef.field_kinds, strict=True)
        )

    def field_external_names(
        self, handle: RecordType | ExceptionType
    ) -> Mapping[str, ExternalName]:
        """Return the renamed fields of *handle*, an exception's base chain applied.

        A field without ``@name``/``@json-name`` is absent from the mapping.
        """
        if isinstance(handle, RecordType):
            typedef = self._require_record_def(handle, caller="field_external_names")
            return dict(typedef.field_external_names)
        return {
            field_name: external
            for _chain_id, typedef in self._exception_chain(
                handle.decl_id, caller="field_external_names"
            )
            for field_name, external in typedef.field_external_names
        }

    def external_name(self, handle: RecordType) -> ExternalName:
        """Return the ``@name``/``@json-name`` spellings of record *handle*'s declaration."""
        return self._require_record_def(handle, caller="external_name").external_name

    def record_doc(self, handle: RecordType) -> str | None:
        """Return a record/member declaration's recognized ``@doc`` prose."""
        return self._require_record_def(handle, caller="record_doc").doc

    def json_fields(self, handle: RecordType | ExceptionType) -> tuple[tuple[str, str, Type], ...]:
        """Return every field of *handle* as ``(declared_name, json_name, field_type)``.

        ``json_name`` is the effective JSON name (``@json-name`` ?? ``@name``
        ?? declared), covering every field — not only renamed ones, unlike
        :meth:`field_external_names`. An exception's base chain is flattened
        in, as for :meth:`exception_fields`.
        """
        renamed = self.field_external_names(handle)
        fields = (
            self.record_fields(handle)
            if isinstance(handle, RecordType)
            else self.exception_fields(handle)
        )
        return tuple(
            (name, renamed[name].json(name) if name in renamed else name, field_type)
            for name, field_type in fields.items()
        )

    def exception_def(self, handle: ExceptionType) -> TypeDef:
        """Return the registered ``TypeDef`` for *handle*.

        Used to read exception hierarchy metadata (``abstract``, ``base``),
        which lives here rather than on the ``ExceptionType`` handle. Raises
        ``KeyError``/``AssertionError`` under the same conditions as
        :meth:`exception_fields`.
        """
        return self._require_exception_def(handle.decl_id, caller="exception_def")

    def _require_exception_def(self, decl_id: DeclId, *, caller: str) -> TypeDef:
        typedef = self._defs.get(decl_id)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for exception identity {decl_id!r}")
        if typedef.kind != "exception":
            raise AssertionError(
                f"{caller} called for identity {decl_id!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'exception'"
            )
        return typedef

    def entries(self) -> tuple[TypeDef, ...]:
        """Return all registered ``TypeDef``s (used for REPL and program table sharing)."""
        return tuple(self._defs.values())

    @property
    def defs(self) -> Mapping[DeclId, TypeDef]:
        """Every registered ``TypeDef``, keyed by its identity, as a read-only view.

        Includes every declaration the table has ever registered, superseded
        or not: a still-registered declaration's field/base references are
        handles naming a specific identity, and a reference to a superseded
        declaration must still resolve to that declaration's own shape.
        """
        return MappingProxyType(self._defs)

    def builtin_declaration(self, name: str) -> TypeDef | None:
        """Return the live registered ``builtin`` declaration named *name*, if any.

        Returns ``None`` for a program that declares no ``builtin`` of that
        name — the caller falls back to the seeded canonical shape, which is
        not itself a declaration.

        Builtin declarations are unique by complete scoped name, so a compile
        unit may contain several paths with this bare name. A live declaration
        outside the standard library overrides the standard declaration; within
        each tier the newest registered match wins. An orphaned declaration
        (:meth:`orphan`) is skipped because it never took effect.
        """
        standard: TypeDef | None = None
        override: TypeDef | None = None
        for decl_id, typedef in self._defs.items():
            if typedef.is_builtin and typedef.name == name and decl_id not in self._orphaned:
                if typedef.module_id.is_standard_library:
                    standard = typedef
                else:
                    override = typedef
        return override if override is not None else standard

    def standard_builtin_declaration(self, name: str) -> TypeDef | None:
        """Return the loaded standard-library source declaration for *name*.

        Host contracts may contain fields whose values are supplied by the
        standard host representation rather than by the selected top-level
        contract declaration. Reserved fallback identities cover the same
        fields when the owning standard-library module is not loaded.
        """
        return self.standard_builtin_declarations().get(name)

    def option_handle(self, argument: Type, *, standard: bool = False) -> EnumType:
        """Return the ``Option[argument]`` handle this program's ``Option`` names.

        The default follows :meth:`builtin_declaration`, so a program's own
        ``builtin enum Option`` owns the values the language itself produces
        (an ``as?`` result). *standard* instead selects the loaded
        standard-library declaration, the identity an ``Option`` nested in a
        fixed host representation carries. Either way a program with no
        registered declaration — one loaded without the standard library —
        falls back to the seeded generic shape.
        """
        declaration = (
            self.standard_builtin_declaration("Option")
            if standard
            else self.builtin_declaration("Option")
        ) or OPTION_TYPE_DEF
        handle = declaration.handle((argument,))
        assert isinstance(handle, EnumType), "Option's declaration must be an enum"
        return handle

    def standard_builtin_declarations(self) -> Mapping[str, TypeDef]:
        """Return all loaded standard-library source builtin declarations."""
        result = self._standard_builtins
        if result is None:
            result = {}
            for decl_id, typedef in self._defs.items():
                if (
                    typedef.is_builtin
                    and typedef.module_id.is_standard_library
                    and decl_id not in self._orphaned
                ):
                    result[typedef.name] = typedef
            self._standard_builtins = result
        return result

    def builtin_declarations(self) -> Mapping[str, TypeDef]:
        """Return every bare name's currently live registered ``builtin`` declaration.

        The all-names counterpart of :meth:`builtin_declaration`: for each
        bare name, it applies the same non-standard-over-standard precedence,
        newest-within-tier tie-break, and orphan filtering. Building
        the host's minting table from this instead of an independent scan is
        how the host and the checker are kept from ever disagreeing about
        which declaration a built-in name denotes (see
        ``lower.lowerer.builtin_nominals_from_declarations``).
        """
        standard: dict[str, TypeDef] = {}
        overrides: dict[str, TypeDef] = {}
        for decl_id, typedef in self._defs.items():
            if typedef.is_builtin and decl_id not in self._orphaned:
                target = standard if typedef.module_id.is_standard_library else overrides
                target[typedef.name] = typedef
        return {**standard, **overrides}

    def host_minted_declaration_ids(self) -> frozenset[DeclId]:
        """Return reserved and loaded-source identities for host-owned resources."""
        cached = self._host_minted_ids
        if cached is None:
            identities = set(HOST_MINTED_PRELUDE_TYPE_IDS)
            for name in HOST_MINTED_PRELUDE_TYPE_NAMES:
                declaration = self.standard_builtin_declaration(name)
                if declaration is not None:
                    identities.add(declaration.decl_node_id)
            cached = frozenset(identities)
            self._host_minted_ids = cached
        return cached

    def nominal_reaches_non_data(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if a non-data type is reachable from *handle* (cycle-safe).

        The non-data types are ``unit``, ``agent``, and function types.
        Declaration-level: *handle*'s declaration reaches one unconditionally
        (``NonDataReachability.reaches_non_data``), or one of its concrete
        ``type_args`` at a relevant parameter position does — see
        :func:`~agm.agl.semantics.analyses.compute_non_data_reachability` for
        why this reproduces the substitute-then-walk answer without ever
        expanding *handle*'s own fields (so it never re-enters a cycle).
        Exceptions carry no ``type_args``, so only the declaration flag
        applies to them.
        """
        caps = self._non_data_reachability()
        decl_id = handle.decl_id
        if decl_id in caps.reaches_non_data:
            return True
        if isinstance(handle, ExceptionType):
            return False
        typedef = self._defs.get(decl_id)
        if typedef is None:
            return False
        relevant = caps.relevant_params.get(decl_id, frozenset())
        return any(
            _reaches_non_data(arg, self)
            for pname, arg in zip(typedef.type_params, handle.type_args)
            if pname in relevant
        )

    def nominal_is_json_convertible(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if *handle* has a JSON representation.

        A record and an exception convert to a JSON object of their fields, an
        enum to ``{"$case": variant, …fields}``, so the only obstacle is a
        non-data leaf somewhere inside — exactly
        :meth:`nominal_reaches_non_data`, negated.
        """
        return not self.nominal_reaches_non_data(handle)

    def _non_data_reachability(self) -> "NonDataReachability":
        if self._non_data_caps is None:
            from agm.agl.semantics.analyses import compute_non_data_reachability

            self._non_data_caps = compute_non_data_reachability(self)
        return self._non_data_caps

    def has_finite_schema(self, t: Type) -> bool:
        """Return ``True`` if every declaration reachable from *t* has a finite closure.

        Thin wrapper over :meth:`first_infinite_declaration`: *t* has a
        finite schema iff no infinite declaration is reachable from it.
        """
        return self.first_infinite_declaration(t) is None

    def first_infinite_declaration(self, t: Type) -> TypeDef | None:
        """Return the first infinite declaration reachable from *t*, or ``None``.

        Walks *t*'s own (finite) type tree for nominal references
        (:func:`~agm.agl.semantics.analyses.nominal_references`), then
        extends to every transitively reachable declaration via the
        (declaration-level, argument-independent) reference graph, breadth-
        first, checking each one's finite-closure flag. Never expands a
        concrete instantiation, so it terminates regardless of how *t*'s
        declarations recurse. Breadth-first (rather than depth-first) and
        ordered deterministically (*t*'s own nominal references in tree
        order, then each further hop sorted by declaration name — never by
        identity, so the reported "culprit" never depends on declaration
        numbering) so that, when *t* itself names an infinite declaration,
        that declaration — the most useful "culprit" for a use-site
        diagnostic — is reported before any declaration reachable only
        through a nested field.
        """
        from agm.agl.semantics.analyses import nominal_references_for_schema

        caps = self._finite_closure_result()
        result_id = bfs_first(
            (
                ref.decl_id
                for ref in nominal_references_for_schema(t, self._defs, caps.relevant_params)
            ),
            lambda decl_id: caps.successors.get(decl_id, frozenset()),
            lambda decl_id: decl_id if decl_id in caps.infinite else None,
            key=self._decl_id_sort_key,
        )
        return None if result_id is None else self._defs.get(result_id)

    def canonical_schema_type(self, t: Type) -> Type:
        """Return *t* with schema-irrelevant nominal type arguments canonicalized.

        Phantom parameters cannot affect a declaration's emitted JSON schema,
        so schema planning must treat instantiations that differ only at those
        positions as the same node. Relevant arguments are canonicalized
        recursively so phantom differences nested inside them are erased too.
        """
        return self._canonical_schema_type(t, self._finite_closure_result().relevant_params)

    def _canonical_schema_type(
        self, t: Type, relevant_params: Mapping[DeclId, frozenset[str]]
    ) -> Type:
        match t:
            case RecordType():
                return RecordType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                    scope_path=t.scope_path,
                    decl_id=t.decl_id,
                )
            case EnumType():
                return EnumType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                    scope_path=t.scope_path,
                    decl_id=t.decl_id,
                )
            case ArrayType(elem=elem):
                return ArrayType(self._canonical_schema_type(elem, relevant_params))
            case DictType(value=value):
                return DictType(self._canonical_schema_type(value, relevant_params))
            case FunctionType(params=params, result=result):
                return FunctionType(
                    params=tuple(self._canonical_schema_type(p, relevant_params) for p in params),
                    result=self._canonical_schema_type(result, relevant_params),
                )
            case (
                ExceptionType()
                | UnitType()
                | TextType()
                | JsonType()
                | BoolType()
                | IntType()
                | DecimalType()
                | BottomType()
                | TypeVarType()
                | InferenceVarType()
            ):
                return t
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _canonical_schema_args(
        self,
        t: RecordType | EnumType,
        relevant_params: Mapping[DeclId, frozenset[str]],
    ) -> tuple[Type, ...]:
        typedef = self._defs.get(t.decl_id)
        if typedef is None:
            return tuple(self._canonical_schema_type(arg, relevant_params) for arg in t.type_args)
        relevant = relevant_params.get(t.decl_id, frozenset())
        result: list[Type] = []
        for pname, arg in zip(typedef.type_params, t.type_args):
            if pname in relevant:
                result.append(self._canonical_schema_type(arg, relevant_params))
            else:
                result.append(UnitType())
        if len(t.type_args) > len(typedef.type_params):
            result.extend(
                self._canonical_schema_type(arg, relevant_params)
                for arg in t.type_args[len(typedef.type_params) :]
            )
        return tuple(result)

    def schema_relevant_type_args(self, t: RecordType | EnumType) -> tuple[Type, ...]:
        """Return the canonical type arguments that should appear in schema identity labels."""
        caps = self._finite_closure_result()
        canonical = self._canonical_schema_type(t, caps.relevant_params)
        if not isinstance(canonical, (RecordType, EnumType)):  # pragma: no cover
            raise AssertionError(f"canonicalized nominal handle became {canonical!r}")
        typedef = self._defs.get(t.decl_id)
        if typedef is None:
            return canonical.type_args
        relevant = caps.relevant_params.get(t.decl_id, frozenset())
        result = [
            arg for pname, arg in zip(typedef.type_params, canonical.type_args) if pname in relevant
        ]
        if len(canonical.type_args) > len(typedef.type_params):
            result.extend(canonical.type_args[len(typedef.type_params) :])
        return tuple(result)

    def schema_relevant_nominal_references(
        self, t: Type
    ) -> tuple[RecordType | EnumType | ExceptionType, ...]:
        """Return nominal references that can affect *t*'s finite schema."""
        from agm.agl.semantics.analyses import nominal_references_for_schema

        caps = self._finite_closure_result()
        result: list[RecordType | EnumType | ExceptionType] = []
        for ref in nominal_references_for_schema(t, self._defs, caps.relevant_params):
            canonical = self._canonical_schema_type(ref, caps.relevant_params)
            result.append(cast(RecordType | EnumType | ExceptionType, canonical))
        return tuple(result)

    def no_finite_schema_message(self, t: Type, *, use: str) -> str | None:
        """Return the use-site diagnostic for *t* if it has no finite JSON schema.

        Returns ``None`` when *t* has a finite schema (:meth:`has_finite_schema`
        is true) — the call site should proceed normally in that case. *use*
        is spliced into one user-facing sentence describing why a schema is
        needed at this use site (e.g. ``"an agent output type"``,
        ``"a cast target"``, ``"a parameter type"``). When the culprit
        declaration IS *t*'s own (e.g. *t* is directly ``Perfect[int]``), only
        *t* is named; when it is reached through a nested field (e.g. a
        non-recursive record containing a ``Perfect[int]`` field), both *t*
        and the culprit declaration's name are mentioned.
        """
        culprit = self.first_infinite_declaration(t)
        if culprit is None:
            return None
        is_own_declaration = (
            isinstance(t, (RecordType, EnumType, ExceptionType))
            and t.decl_id == culprit.decl_node_id
        )
        if is_own_declaration:
            return (
                f"type '{t!r}' cannot be used as {use}: its recursive instantiations "
                "never close, so it has no finite JSON schema."
            )
        return (
            f"type '{t!r}' cannot be used as {use}: it contains "
            f"'{qualified_decl_name(culprit)}', whose recursive instantiations never "
            "close, so it has no finite JSON schema."
        )

    def json_representation_obstacle(self, t: Type) -> str | None:
        """Return why *t* has no JSON representation, as a diagnostic clause.

        ``None`` when *t* converts (:func:`is_json_convertible` is true) or
        when nothing more specific than "this type does not convert" can be
        said — an unresolved inference variable, say, whose real problem is
        inference rather than representation. Otherwise a clause naming the
        culprit, for a caller to splice into its own sentence: a non-data type
        reached structurally, the declaration field that carries one, or the
        type variable that may stand for one. Shaped like
        :meth:`no_finite_schema_message`, whose culprit search this mirrors.
        """
        if is_json_convertible(t, self):
            return None
        leaf = _first_non_data_leaf(t)
        if leaf is not None:
            return f"'{leaf!r}' has no JSON representation"
        culprit = self.first_non_data_field(t)
        if culprit is not None:
            typedef, field_name, field_type = culprit
            return (
                f"field '{field_name}' of '{qualified_decl_name(typedef)}' has type "
                f"'{field_type!r}', which has no JSON representation"
            )
        type_vars = sorted(free_type_vars(t))
        if type_vars:
            return (
                f"type variable '{type_vars[0]}' may stand for a type with no JSON representation"
            )
        return None

    def first_non_data_field(self, t: Type) -> tuple[TypeDef, str, Type] | None:
        """Return the declaration field that costs *t* its JSON representation.

        The result is ``(declaration, field name, that field's declared
        type)``. Breadth-first from *t*'s own nominal references, so the
        shallowest declaration carrying a non-data field is reported — the
        most useful culprit for a use-site diagnostic — before one reachable
        only through further hops. Follows a declaration's affected field
        references, an exception's ``extends`` base (whose fields are
        inherited) and its affected descendants (a value statically typed as
        the ancestor may hold one at runtime). Never expands an instantiation,
        so it terminates however the declarations recurse. ``None`` when no
        reachable declaration is to blame.
        """
        from agm.agl.semantics.analyses import nominal_references

        def culprit(decl_id: DeclId) -> tuple[TypeDef, str, Type] | None:
            typedef = self._defs.get(decl_id)
            if typedef is None:  # pragma: no cover
                # Unreachable by construction: every identity enqueued was
                # first confirmed to reach a non-data type, which a dangling
                # (never-registered) declaration never does. Kept as a
                # defensive guard, matching the dangling-reference handling in
                # the fixpoints themselves, in case that ever stops holding.
                return None
            direct = self._own_non_data_field(typedef)
            return None if direct is None else (typedef, *direct)

        def successors(decl_id: DeclId) -> set[DeclId]:
            typedef = self._defs.get(decl_id)
            return set() if typedef is None else self._affected_successors(decl_id, typedef)

        return bfs_first(
            (ref.decl_id for ref in nominal_references(t) if self.nominal_reaches_non_data(ref)),
            successors,
            culprit,
            key=self._decl_id_sort_key,
        )

    def _own_non_data_field(self, typedef: TypeDef) -> tuple[str, Type] | None:
        """Return *typedef*'s first own field whose type structurally reaches non-data."""
        from agm.agl.semantics.analyses import field_templates

        for field_name, template in field_templates(typedef, self._defs):
            if _first_non_data_leaf(template) is not None:
                return field_name, template
        return None

    def _affected_successors(self, decl_id: DeclId, typedef: TypeDef) -> set[DeclId]:
        """Return the declarations *typedef* reaches non-data through.

        A field reference is a HANDLE, so it is tested with
        :meth:`nominal_reaches_non_data`, which also consults the concrete
        type arguments — ``Box[agent]`` is affected while ``Box`` itself is
        not. An exception's ``extends`` base and its descendants are bare
        declaration identities carrying no arguments, so for them the
        declaration-level ``flags`` set is the whole answer. The two idioms
        below are therefore not interchangeable.
        """
        from agm.agl.semantics.analyses import field_templates, nominal_references

        flags = self._non_data_reachability().reaches_non_data
        result: set[DeclId] = set()
        for _field_name, template in field_templates(typedef, self._defs):
            for ref in nominal_references(template):
                if self.nominal_reaches_non_data(ref):
                    result.add(ref.decl_id)
        if typedef.kind == "exception":
            if typedef.base is not None and typedef.base in flags:
                result.add(typedef.base)
            result.update(
                child_id
                for child_id, child in self._defs.items()
                if child.kind == "exception" and child.base == decl_id and child_id in flags
            )
        return result

    def _decl_id_sort_key(self, decl_id: DeclId) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        """Resolve *decl_id* against this table for :func:`decl_id_sort_key`."""
        return decl_id_sort_key(self._defs, decl_id)

    def _finite_closure_result(self) -> "FiniteClosure":
        if self._finite_closure is None:
            from agm.agl.semantics.analyses import compute_finite_closure

            self._finite_closure = compute_finite_closure(self)
        return self._finite_closure

    def merge_from(self, other: "TypeTable") -> None:
        """Copy every entry and name-index mapping from *other* into this table.

        Used to carry accumulated declarations across REPL entries (and to
        seed a fresh per-entry environment from the session's persisted
        state). *other* is treated as authoritative: a def already present
        under the same identity is overwritten, and *other*'s name index
        entries overwrite this table's own — mirroring the last-write-wins
        semantics already used to seed the embedded type dict (``_types``).
        Callers that seed the SAME live table from the SAME source more than
        once within one check (see
        :meth:`~agm.agl.typecheck.env.TypeEnvironment.seed_from`'s
        ``merge_type_table``) must skip a second, now-stale call instead of
        relying on this method to detect it — a later declaration under a
        name path this table already binds to a different identity is
        expected to take the name over unconditionally, so there is no
        general way to tell that apart from a stale re-seed here.

        Skips the write (and the resulting cache invalidation) entirely when
        the incoming def is identical to the one already registered under
        that identity, since no cached substitution can be stale in that case.
        """
        for decl_id, typedef in other._defs.items():
            if self._defs.get(decl_id) == typedef:
                continue
            self._defs[decl_id] = typedef
            self._methods.pop(decl_id, None)
            self._invalidate_cache_for(decl_id)
        for name_key, decl_id in other._name_index.items():
            self._name_index[name_key] = decl_id
        # Orphan status travels with the declaration: a session seeds a fresh
        # table from its accumulated one on every entry, so a declaration
        # orphaned once must stay orphaned for the rest of the session.
        self._orphaned |= other._orphaned
        for decl_id, methods in other._methods.items():
            for candidates in methods.values():
                for method in candidates.values():
                    self._put_method(decl_id, method)
        for constructor, methods in other._builtin_methods.items():
            for candidates in methods.values():
                for method in candidates.values():
                    self.register_builtin_method(constructor, method)


def decl_def_sort_key(typedef: TypeDef) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Deterministic sort key for a ``TypeDef`` (module, scope path, then name).

    Name-based, never identity-based, so a "first"/"culprit" declaration
    chosen by sorting a set of ``TypeDef``s never depends on declaration
    numbering (AST node id order).
    """
    return (typedef.module_id.segments, typedef.scope_path, typedef.name)


def decl_id_sort_key(
    defs: Mapping[DeclId, TypeDef], decl_id: DeclId
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Deterministic sort key for *decl_id*, resolved to its declaration in *defs*.

    Every SCC/BFS traversal that has to order declaration identities sorts by
    declaration name (:func:`decl_def_sort_key`) rather than by identity, so a
    "first"/"culprit" declaration chosen from a fixpoint never depends on
    declaration numbering. A *decl_id* absent from *defs* (a dangling
    reference — an internal-invariant violation the fixpoints handle
    defensively rather than assume away) sorts after every named declaration,
    using the raw identity only to keep multiple dangling entries mutually
    ordered.
    """
    typedef = defs.get(decl_id)
    if typedef is None:  # pragma: no cover
        return ((), (), f"￿<dangling:{decl_id}>")
    return decl_def_sort_key(typedef)


def qualified_decl_name(typedef: TypeDef) -> str:
    """Return *typedef*'s user-facing name, module-qualified where a reader needs it.

    Bare names from an imported module can otherwise be ambiguous; the
    ``module::name`` convention matches ``RecordType``/``EnumType``'s own
    ``__repr__``, and shares the same bare/qualified decision
    (:func:`~agm.agl.semantics.types.spells_bare`): entry-module declarations
    and the shipped standard library's own built-in names are spelled bare
    because that is how every reader writes them.
    """
    module, scope_path, name = typedef.module_id, typedef.scope_path, typedef.name
    scoped = "::".join((*scope_path, name))
    if spells_bare(module, name):
        return scoped
    return f"{module.display()}::{scoped}"


def _first_non_data_leaf(t: Type) -> Type | None:
    """Return the first non-data type reachable through *t*'s own structure.

    Structural only: recurses through ``array``/``dict`` but stops at a nominal
    handle, whose own fields are a declaration-level question answered by
    :meth:`TypeTable.first_non_data_field` instead.
    """
    if isinstance(t, (UnitType, FunctionType)):
        return t
    if isinstance(t, (RecordType, EnumType, ExceptionType)):
        return None
    for child in type_children(t):
        leaf = _first_non_data_leaf(child)
        if leaf is not None:
            return leaf
    return None


def _reaches_non_data(t: Type, table: TypeTable) -> bool:
    """True if ``t`` is, or transitively contains, a non-data type.

    The non-data types are function and ``unit``: function and agent
    values are opaque / identity-only, and ``unit`` has a single value carrying
    nothing.  An array, dict, record, enum, or exception that transitively
    holds one is therefore itself affected.  ``t`` is always a finite tree
    (array/dict wrapping is structural, not nominal), so recursing through
    ``ArrayType``/``DictType`` always terminates; a record/enum/exception
    handle instead defers to :meth:`TypeTable.nominal_reaches_non_data`, which
    consults a precomputed declaration-level fixpoint rather than re-walking
    the handle's own fields — the type declarations themselves may be
    recursive, but this function never re-enters them.
    """
    match t:
        case FunctionType() | UnitType():
            return True
        case ArrayType():
            return _reaches_non_data(t.elem, table)
        case DictType():
            return _reaches_non_data(t.value, table)
        case RecordType() | EnumType() | ExceptionType():
            return table.nominal_reaches_non_data(t)
        case (
            TextType()
            | JsonType()
            | BoolType()
            | IntType()
            | DecimalType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def comparable_types(left: Type, right: Type, table: TypeTable) -> bool:
    """Return ``True`` if ``left`` and ``right`` may be compared.

    Equality (``=``, ``!=``) and ordering comparisons require both operands to
    have the **same** type after the single ``int → decimal`` widening.  Unlike
    :func:`~agm.agl.semantics.types.is_assignable`, ``json`` does **not** absorb
    JSON-shaped scalars here: ``json = json`` is allowed but ``json`` vs any
    non-``json`` type is a static error.  Records/enums/exceptions compare only
    with their own exact type.

    ``FunctionType`` and ``UnitType`` operands are
    NON-comparable — using ``=``/``!=``/``<`` on them is a static error.
    This rule is **transitive**: an ``array``, ``dict``, ``record``, ``enum``, or
    ``exception`` that (at any depth) contains a function or ``unit``
    value likewise has no equality and cannot be compared with ``=``/``!=``.
    ``table`` resolves record/enum field shapes for that transitive walk.
    """
    # Function/unit values — and any container/record/enum that transitively
    # holds one — have no value equality.
    if _reaches_non_data(left, table) or _reaches_non_data(right, table):
        return False
    # Bare type variables and the bottom type are never comparable here (the
    # checker additionally rejects bare type variables at the comparison site).
    if isinstance(left, (BottomType, TypeVarType, InferenceVarType)) or isinstance(
        right, (BottomType, TypeVarType, InferenceVarType)
    ):
        return False
    if left == right:
        return True
    # The only cross-type comparison is numeric int↔decimal (either direction).
    numeric = (IntType, DecimalType)
    return isinstance(left, numeric) and isinstance(right, numeric)


# ---------------------------------------------------------------------------
# JSON convertibility and cast classification
# ---------------------------------------------------------------------------


def is_json_convertible(t: Type, table: TypeTable) -> bool:
    """Return ``True`` if ``t`` has a JSON representation.

    The scalars (``text``/``json``/``bool``/``int``/``decimal``) convert
    directly; an ``array``/``dict`` converts iff its element/value type does;
    a record or exception converts to a JSON object of its fields and an enum
    to ``{"$case": variant, …fields}``, so a nominal converts iff no non-data
    type is reachable from its declaration
    (:meth:`TypeTable.nominal_is_json_convertible`). The non-data types —
    ``unit``, ``agent``, and function types — have no representation at all.

    A free type variable is never convertible, its own arm here and, for a
    nominal, in its type arguments: casts are compiled once and type arguments
    are erased, so a ``T`` later instantiated with ``agent`` would otherwise
    reach the conversion at runtime. Note the deliberate asymmetry with the
    declaration-level fixpoint, which correctly treats a type variable in a
    field template as *not* a problem — that is what its relevant-parameter
    analysis is for.

    This is what an explicit ``as json`` cast accepts. It is deliberately
    wider than :func:`~agm.agl.semantics.types.is_json_shaped` (what may
    *inhabit* a ``json`` slot) and than
    :func:`~agm.agl.semantics.types.is_scalar_json_shaped` (what an *implicit*
    coercion absorbs); all three answer different questions.
    """
    match t:
        case TextType() | JsonType() | BoolType() | IntType() | DecimalType():
            return True
        case ArrayType():
            return is_json_convertible(t.elem, table)
        case DictType():
            return is_json_convertible(t.value, table)
        case RecordType() if t.decl_id in HOST_MINTED_PRELUDE_TYPE_IDS:
            return False
        case ExceptionType():
            return table.nominal_is_json_convertible(t)
        case RecordType() | EnumType():
            return table.nominal_is_json_convertible(t) and not any(
                contains_type_var(arg) for arg in t.type_args
            )
        case UnitType() | FunctionType() | BottomType() | TypeVarType() | InferenceVarType():
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def is_assignable_in(table: TypeTable, value_type: Type, target_type: Type) -> bool:
    """Return whether a value is assignable to a target in *table*'s nominal context.

    The pure :func:`semantics.types.is_assignable` rules apply unchanged.  In
    addition, a member record is assignable to an enum when it occurs in that
    enum instantiation's declared member set, and an exception is assignable
    to any exception in its base chain. This is deliberately a top-level,
    directed relation: containers remain invariant. An enum is assignable to
    another enum exactly when its constructor set is a subset of the target's.
    """
    if is_assignable(value_type, target_type):
        return True
    if (
        isinstance(value_type, RecordType)
        and isinstance(target_type, EnumType)
        and value_type in table.enum_members(target_type)
    ):
        return True
    if isinstance(value_type, EnumType) and isinstance(target_type, EnumType):
        if (
            table.get_by_id(value_type.decl_id) is None
            or table.get_by_id(target_type.decl_id) is None
        ):
            return False
        return table.enum_is_subset(value_type, target_type)
    if isinstance(value_type, ExceptionType) and isinstance(target_type, ExceptionType):
        return table.is_exception_ancestor(target_type.decl_id, value_type.decl_id)
    return False


def json_cast_hint(value_type: Type, target_type: Type, table: TypeTable) -> str:
    """Return a diagnostic clause naming an explicit ``as json`` cast, or ``""``.

    Directional: it only ever fires in the *value → json* direction, when
    ``target_type`` is ``json`` and ``value_type`` is accepted by an explicit
    ``as json`` cast (:func:`is_json_convertible`) but not implicitly absorbed
    by assignment (:func:`~agm.agl.semantics.types.is_scalar_json_shaped`) —
    an ``array``/``dict`` (of JSON-shaped elements) or a JSON-convertible
    record/enum/exception. It never fires when ``value_type`` is ``json`` and
    ``target_type`` is something else: an explicit ``as json`` cast is not the
    fix for *that* mismatch.
    """
    if not isinstance(target_type, JsonType):
        return ""
    if is_scalar_json_shaped(value_type):
        return ""
    if not is_json_convertible(value_type, table):
        return ""
    return " Only scalar values are implicitly absorbed into json; use an explicit 'as json' cast."


def cast_classification(source: Type, target: Type, table: TypeTable) -> CastKind:
    """Classify a cast from source to target type.

    Returns the CastKind for the (source, target) pair. ``table`` resolves the
    declaration-level facts a ``json`` target needs (see
    :func:`is_json_convertible`).
    """
    # Bottom is a valid source because a raise expression never reaches the
    # conversion. Other non-data sources and all non-data targets are invalid.
    if isinstance(source, (UnitType, FunctionType)) or isinstance(
        target, (UnitType, FunctionType, BottomType)
    ):
        return CastKind.STATIC_ERROR
    # Exception hierarchy casts: same declaration is a no-op, an ancestor
    # (including the root) is an upcast, a descendant is a runtime-checked
    # downcast; unrelated exceptions are a static error.
    if isinstance(source, ExceptionType) and isinstance(target, ExceptionType):
        if source.decl_id == target.decl_id:
            return CastKind.TOTAL_NOOP
        if table.is_exception_ancestor(target.decl_id, source.decl_id):
            return CastKind.IDENTITY_UPCAST
        if table.is_exception_ancestor(source.decl_id, target.decl_id):
            return CastKind.NOMINAL_DOWNCAST
        return CastKind.STATIC_ERROR
    # Every other source (text/json/etc.) casting to an exception is a static
    # error: exception construction is never implicit or fallible-decoded.
    if isinstance(target, ExceptionType):
        return CastKind.STATIC_ERROR

    # Nominal membership casts preserve the constructor record. Enum widening
    # is statically established; narrowing to a member or an overlapping enum
    # needs a runtime constructor-identity check.
    if isinstance(source, RecordType) and isinstance(target, EnumType):
        if source in table.enum_members(target):
            return CastKind.IDENTITY_UPCAST
        return CastKind.STATIC_ERROR
    if isinstance(source, EnumType) and isinstance(target, RecordType):
        if target in table.enum_members(source):
            return CastKind.NOMINAL_DOWNCAST
        return CastKind.STATIC_ERROR
    if isinstance(source, EnumType) and isinstance(target, EnumType):
        shared = table.shared_enum_members(source, target)
        if not shared:
            return CastKind.STATIC_ERROR
        if len(shared) == len(table.enum_members(source)):
            return CastKind.TOTAL_NOOP
        return CastKind.NOMINAL_DOWNCAST

    # Handle is_assignable cases first (no-op / widen / json-absorb).
    # Note: is_assignable(X, TextType) is true only when X is TextType itself
    # (no implicit widening to text), so the only assignable-to-text case is noop.
    # is_assignable(X, JsonType) is true only for scalar json-shaped types.
    if is_assignable(source, target):
        if isinstance(target, JsonType):
            # json → json: noop; all other json-shaped sources → canonicalize
            if isinstance(source, JsonType):
                return CastKind.TOTAL_NOOP
            return CastKind.TOTAL_JSON
        # All other assignable cases are no-ops (including int→decimal widen,
        # same-type identity, etc.)
        return CastKind.TOTAL_NOOP

    # Now source is NOT assignable to target.
    _text_or_json = (TextType, JsonType)

    if isinstance(target, TextType):
        # Every data value renders to text. Non-data sources (unit/function)
        # are filtered at the top, and json-shaped/exact-type sources are handled by
        # the is_assignable block above, so any source reaching here is a renderable
        # data type (json/bool/int/decimal/array/dict/record/enum/exception).
        return CastKind.TOTAL_RENDER

    if isinstance(target, JsonType):
        # Scalar json-shaped sources are assignable to json (handled above), so
        # anything reaching here needs the full rule: every type with a JSON
        # representation converts, and nothing else does.
        if is_json_convertible(source, table):
            return CastKind.TOTAL_JSON
        return CastKind.STATIC_ERROR

    if isinstance(target, (BoolType, IntType, DecimalType)):
        # decimal → int is a narrowing cast (fallible); text/json → numeric is fallible.
        if isinstance(source, _text_or_json) or (
            isinstance(target, IntType) and isinstance(source, DecimalType)
        ):
            return CastKind.FALLIBLE
        return CastKind.STATIC_ERROR

    if isinstance(target, (ArrayType, DictType, RecordType, EnumType)):
        if isinstance(source, _text_or_json):
            return CastKind.FALLIBLE
        return CastKind.STATIC_ERROR

    # All target types are covered above; this is a safety fallback.
    return CastKind.STATIC_ERROR  # pragma: no cover


def parse_classification(target: Type, table: TypeTable) -> CastKind:
    """Classify a ``std/value::parse``/``try-parse`` conversion from text to *target*.

    Same as :func:`cast_classification` from a ``text`` source, except a
    ``json`` target is ``FALLIBLE`` (parses the text) rather than
    ``TOTAL_JSON`` (wraps the text as a JSON string, as ``text as json`` does).
    """
    if isinstance(target, JsonType):
        return CastKind.FALLIBLE
    return cast_classification(TextType(), target, table)


# ---------------------------------------------------------------------------
# Prelude type shapes — the single source of truth for built-in nominal types
#
# These ``TypeDef`` literals are the canonical shapes for AgL's built-in
# prelude types (``ExecResult``, ``ParsePolicy``, ``Agent``, ``OutputContract``,
# ``OutputContractOption``, ``AgentRequest``, ``SessionTransport``, ``Session``,
# ``SessionStats``, ``SessionError``) and the generic ``Option``
# template.  ``create_seeded_type_table``, the scope resolver's builtin
# constructor-candidate seeding, ``TypeEnvironment`` init seeding, and builtin
# shape validation in the type builder all read these same literals — there
# is exactly one definition of each prelude shape.
# ---------------------------------------------------------------------------


def _standard(fields: tuple[tuple[str, Type], ...]) -> tuple[ParamZone, ...]:
    """Return one ``ParamZone.STANDARD`` per entry of *fields*."""
    return (ParamZone.STANDARD,) * len(fields)


def _builtin_enum_defs(
    name: str,
    variants: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...],
    *,
    type_params: tuple[str, ...] = (),
    module_id: ModuleId = RESERVED_ID,
) -> tuple[TypeDef, tuple[TypeDef, ...]]:
    """Build canonical enum and scoped record-member definitions for the prelude."""
    scope_path = (name,)
    member_defs = tuple(
        TypeDef(
            kind="record",
            name=member_name,
            module_id=module_id,
            scope_path=scope_path,
            type_params=tuple(
                param
                for param in type_params
                if any(param in free_type_vars(field_type) for _field, field_type in fields)
            ),
            fields=fields,
            field_kinds=_standard(fields),
            decl_node_id=require_reserved_enum_member_id(name, member_name),
        )
        for member_name, fields in variants
    )
    members = tuple(
        cast(RecordType, member.handle(tuple(TypeVarType(param) for param in member.type_params)))
        for member in member_defs
    )
    return (
        TypeDef(
            kind="enum",
            name=name,
            module_id=module_id,
            type_params=type_params,
            members=members,
        ),
        member_defs,
    )


_PARSE_POLICY_DEF, _PARSE_POLICY_MEMBER_DEFS = _builtin_enum_defs(
    "ParsePolicy",
    (("Abort", ()), ("Retry", (("n", IntType()),))),
)
_AGENT_DEF, _AGENT_MEMBER_DEFS = _builtin_enum_defs(
    "Agent",
    (
        ("AgentCommand", (("command", TextType()),)),
        ("AgentClaude", (("model", TextType()), ("thinking", TextType()))),
        ("AgentCodex", (("model", TextType()), ("thinking", TextType()))),
        (
            "AgentPi",
            (("provider", TextType()), ("model", TextType()), ("thinking", TextType())),
        ),
    ),
)
_SESSION_TRANSPORT_DEF, _SESSION_TRANSPORT_MEMBER_DEFS = _builtin_enum_defs(
    "SessionTransport", (("Cli", ()), ("Rpc", ()))
)

_OUTPUT_CONTRACT_OPTION_DEF, _OUTPUT_CONTRACT_OPTION_MEMBER_DEFS = _builtin_enum_defs(
    "OutputContractOption",
    (
        ("None", ()),
        (
            "Some",
            (
                (
                    "value",
                    RecordType(
                        name="OutputContract",
                        module_id=RESERVED_ID,
                        decl_id=_reserved_id("OutputContract"),
                    ),
                ),
            ),
        ),
    ),
)
_OPTION_DEF, _OPTION_MEMBER_DEFS = _builtin_enum_defs(
    "Option",
    (("None", ()), ("Some", (("value", TypeVarType("T")),))),
    type_params=("T",),
)
_OPTIONAL_DEFAULT_DEF = TypeDef(
    kind="record",
    name="Default",
    module_id=RESERVED_ID,
    scope_path=("Optional",),
    decl_node_id=require_reserved_enum_member_id("Optional", "Default"),
)
_OPTIONAL_DEF = TypeDef(
    kind="enum",
    name="Optional",
    module_id=RESERVED_ID,
    type_params=("T",),
    members=(
        RecordType(
            name="Some",
            type_args=(TypeVarType("T"),),
            module_id=RESERVED_ID,
            scope_path=("Option",),
            decl_id=require_reserved_enum_member_id("Option", "Some"),
        ),
        RecordType(
            name="None",
            module_id=RESERVED_ID,
            scope_path=("Option",),
            decl_id=require_reserved_enum_member_id("Option", "None"),
        ),
        RecordType(
            name="Default",
            module_id=RESERVED_ID,
            scope_path=("Optional",),
            decl_id=require_reserved_enum_member_id("Optional", "Default"),
        ),
    ),
)


def _with_reserved_ids(defs: Mapping[str, TypeDef]) -> Mapping[str, TypeDef]:
    """Stamp each canonical entry with the reserved identity its own key names.

    A prelude shape is keyed by the built-in name it defines, so its
    declaration identity is implied by that key; deriving it here rather than
    repeating ``_reserved_id("Name")`` in every entry keeps a shape from ever
    carrying another name's identity.
    """
    return {
        name: replace(typedef, decl_node_id=_reserved_id(name)) for name, typedef in defs.items()
    }


#: Reused as a walrus target below, one entry at a time, to derive each
#: shape's ``field_kinds`` from its own ``fields`` without repeating them;
#: annotated so each narrower assignment type-checks against this declared
#: type rather than the first literal's.
_fields: tuple[tuple[str, Type], ...]

_PRELUDE_SHAPES: Mapping[str, TypeDef] = {
    "ExecResult": TypeDef(
        kind="record",
        name="ExecResult",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("stdout", TextType()),
                ("exit-code", IntType()),
                ("stderr", TextType()),
                ("timed-out", BoolType()),
            )
        ),
        field_kinds=_standard(_fields),
    ),
    "ParsePolicy": _PARSE_POLICY_DEF,
    "Agent": _AGENT_DEF,
    "OutputContract": TypeDef(
        kind="record",
        name="OutputContract",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("target-type", TextType()),
                ("codec-name", TextType()),
                ("strict-json", JsonType()),
                ("format-instructions", TextType()),
                ("json-schema", JsonType()),
                ("structured-exec", BoolType()),
            )
        ),
        field_kinds=_standard(_fields),
    ),
    "OutputContractOption": _OUTPUT_CONTRACT_OPTION_DEF,
    "AgentRequest": TypeDef(
        kind="record",
        name="AgentRequest",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                (
                    "agent",
                    EnumType(name="Agent", module_id=RESERVED_ID, decl_id=_reserved_id("Agent")),
                ),
                ("prompt", TextType()),
                (
                    "target-type",
                    standard_option_type(TextType()),
                ),
                (
                    "format-instructions",
                    standard_option_type(TextType()),
                ),
                (
                    "json-schema",
                    standard_option_type(JsonType()),
                ),
                ("attempt", IntType()),
                (
                    "previous-error",
                    standard_option_type(TextType()),
                ),
                ("metadata", JsonType()),
            )
        ),
        field_kinds=_standard(_fields),
    ),
    "SessionTransport": _SESSION_TRANSPORT_DEF,
    "Session": TypeDef(
        kind="record",
        name="Session",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("id", TextType()),
                (
                    "agent",
                    EnumType(name="Agent", module_id=RESERVED_ID, decl_id=_reserved_id("Agent")),
                ),
                (
                    "transport",
                    EnumType(
                        name="SessionTransport",
                        module_id=RESERVED_ID,
                        decl_id=_reserved_id("SessionTransport"),
                    ),
                ),
            )
        ),
        field_kinds=_standard(_fields),
    ),
    "SessionStats": TypeDef(
        kind="record",
        name="SessionStats",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("input-tokens", IntType()),
                ("output-tokens", IntType()),
                ("cost", DecimalType()),
                ("context-percent", DecimalType()),
            )
        ),
        field_kinds=_standard(_fields),
    ),
    "SessionError": TypeDef(
        kind="exception",
        name="SessionError",
        module_id=RESERVED_ID,
        fields=(_fields := (("operation", TextType()),)),
        base=_reserved_id("Exception"),
        field_kinds=_standard(_fields),
    ),
}

BUILTIN_PRELUDE_TYPE_DEFS: Mapping[str, TypeDef] = _with_reserved_ids(_PRELUDE_SHAPES)

# Generic ``Option`` template under the reserved sentinel (type parameter ``T``,
# variants ``None``/``Some(value: T)``), matching the shape of the concrete
# ``Option[text]``/``Option[json]`` prelude constants, so a program loaded
# without the standard library can still resolve its member set on
# ``Option`` handles.
OPTION_TYPE_DEF = replace(_OPTION_DEF, decl_node_id=_reserved_id("Option"))
OPTIONAL_TYPE_DEF = replace(_OPTIONAL_DEF, decl_node_id=_reserved_id("Optional"))
BUILTIN_PRELUDE_MEMBER_TYPE_DEFS: Mapping[DeclId, TypeDef] = {
    member.decl_node_id: member
    for member in (
        *_PARSE_POLICY_MEMBER_DEFS,
        *_AGENT_MEMBER_DEFS,
        *_OUTPUT_CONTRACT_OPTION_MEMBER_DEFS,
        *_SESSION_TRANSPORT_MEMBER_DEFS,
        *_OPTION_MEMBER_DEFS,
        _OPTIONAL_DEFAULT_DEF,
    )
}


def source_nominal_decl_id(
    module_id: ModuleId, scope_path: tuple[str, ...], name: str, node_id: int
) -> int:
    """Return the semantic identity of a source nominal declaration.

    Every parsed declaration retains its parser node identity. Reserved
    identities belong only to seeded fallback definitions used when a program
    loads no source declaration for a host-known type.
    """
    return node_id


def source_enum_member_decl_id(
    module_id: ModuleId,
    scope_path: tuple[str, ...],
    enum_name: str,
    member_name: str,
    node_id: int,
    *,
    is_builtin: bool,
) -> int:
    """Return a source inline enum member's parser node identity."""
    return node_id


# ---------------------------------------------------------------------------
# Built-in exception shapes — the single source of truth for every entry of
# ``semantics.types.BUILTIN_EXCEPTIONS``.  ``fields`` holds each exception's
# OWN fields only (the root's ``message`` is NOT repeated on every concrete
# exception — see :meth:`TypeTable.exception_fields`, which flattens the
# ``base`` chain on demand).  ``field_kinds`` is likewise own-fields-only: the
# root's ``message`` is NAMED_ONLY, every other built-in exception field is
# STANDARD — see :meth:`TypeTable.field_kinds`.
# ---------------------------------------------------------------------------

_EXCEPTION_ROOT_ID: DeclId = _reserved_id("Exception")

# Shared field shape for CastError and ValueParseError: both report a failed
# type-directed conversion by its source/target type names and the raw value.
_CAST_LIKE_FIELDS: tuple[tuple[str, Type], ...] = (
    ("source-type", TextType()),
    ("target-type", TextType()),
    ("raw", TextType()),
)


_EXCEPTION_SHAPES: Mapping[str, TypeDef] = {
    "Exception": TypeDef(
        kind="exception",
        name="Exception",
        module_id=RESERVED_ID,
        fields=(_fields := (("message", TextType()),)),
        abstract=True,
        field_kinds=(ParamZone.NAMED_ONLY,) * len(_fields),
    ),
    "AgentCallError": TypeDef(
        kind="exception",
        name="AgentCallError",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                (
                    "agent",
                    EnumType(name="Agent", module_id=RESERVED_ID, decl_id=_reserved_id("Agent")),
                ),
                ("cause", TextType()),
                ("metadata", JsonType()),
            )
        ),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "AgentParseError": TypeDef(
        kind="exception",
        name="AgentParseError",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                (
                    "agent",
                    EnumType(name="Agent", module_id=RESERVED_ID, decl_id=_reserved_id("Agent")),
                ),
                ("target-type", TextType()),
                ("expected-schema", JsonType()),
                ("raw", TextType()),
                ("normalized-raw", TextType()),
                ("validation-errors", JsonType()),
                ("attempts", IntType()),
                ("metadata", JsonType()),
            )
        ),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "ExecError": TypeDef(
        kind="exception",
        name="ExecError",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("command", TextType()),
                ("exit-code", IntType()),
                ("stdout", TextType()),
                ("stderr", TextType()),
                ("timed-out", BoolType()),
            )
        ),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    # ``python_type`` is the raising Python exception's class name, or empty for
    # a contract violation (no Python exception was involved).
    "ExternError": TypeDef(
        kind="exception",
        name="ExternError",
        module_id=RESERVED_ID,
        fields=(_fields := (("function", TextType()), ("python-type", TextType()))),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "MaxIterationsExceeded": TypeDef(
        kind="exception",
        name="MaxIterationsExceeded",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("limit", IntType()),
                ("condition", TextType()),
                ("last-condition-value", BoolType()),
                ("metadata", JsonType()),
            )
        ),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "MatchError": TypeDef(
        kind="exception",
        name="MatchError",
        module_id=RESERVED_ID,
        fields=(_fields := (("scrutinee-type", TextType()), ("scrutinee", JsonType()))),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "IndexError": TypeDef(
        kind="exception",
        name="IndexError",
        module_id=RESERVED_ID,
        fields=(_fields := (("index", IntType()), ("length", IntType()))),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "KeyError": TypeDef(
        kind="exception",
        name="KeyError",
        module_id=RESERVED_ID,
        fields=(_fields := (("key", TextType()),)),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "TypeError": TypeDef(
        kind="exception",
        name="TypeError",
        module_id=RESERVED_ID,
        base=_EXCEPTION_ROOT_ID,
    ),
    "ArithmeticError": TypeDef(
        kind="exception",
        name="ArithmeticError",
        module_id=RESERVED_ID,
        fields=(_fields := (("operation", TextType()),)),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    # Statically prevented by scope/typecheck (assignment to immutable bindings
    # and undeclared names), but still listed as catchable runtime exceptions
    # for any runtime paths that bypass the static passes.
    "UndefinedVariableError": TypeDef(
        kind="exception",
        name="UndefinedVariableError",
        module_id=RESERVED_ID,
        fields=(_fields := (("name", TextType()),)),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "ImmutableBindingError": TypeDef(
        kind="exception",
        name="ImmutableBindingError",
        module_id=RESERVED_ID,
        fields=(_fields := (("name", TextType()), ("operation", TextType()))),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "Abort": TypeDef(
        kind="exception",
        name="Abort",
        module_id=RESERVED_ID,
        base=_EXCEPTION_ROOT_ID,
    ),
    # AgL: RecursionError raised when the call-depth limit is exceeded.
    "RecursionError": TypeDef(
        kind="exception",
        name="RecursionError",
        module_id=RESERVED_ID,
        fields=(_fields := (("limit", IntType()),)),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "CastError": TypeDef(
        kind="exception",
        name="CastError",
        module_id=RESERVED_ID,
        fields=_CAST_LIKE_FIELDS,
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_CAST_LIKE_FIELDS),
    ),
    "ValueParseError": TypeDef(
        kind="exception",
        name="ValueParseError",
        module_id=RESERVED_ID,
        fields=_CAST_LIKE_FIELDS,
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_CAST_LIKE_FIELDS),
    ),
    "JsonParseError": TypeDef(
        kind="exception",
        name="JsonParseError",
        module_id=RESERVED_ID,
        fields=(_fields := (("raw", TextType()),)),
        base=_EXCEPTION_ROOT_ID,
        field_kinds=_standard(_fields),
    ),
    "RangeError": TypeDef(
        kind="exception",
        name="RangeError",
        module_id=RESERVED_ID,
        base=_EXCEPTION_ROOT_ID,
    ),
    "CyclicValueError": TypeDef(
        kind="exception",
        name="CyclicValueError",
        module_id=RESERVED_ID,
        base=_EXCEPTION_ROOT_ID,
    ),
}

BUILTIN_EXCEPTION_TYPE_DEFS: Mapping[str, TypeDef] = _with_reserved_ids(_EXCEPTION_SHAPES)


def create_seeded_type_table() -> TypeTable:
    """Return a fresh ``TypeTable`` pre-populated with built-in defs.

    Registers ``BUILTIN_PRELUDE_TYPE_DEFS`` (``ExecResult``, ``ParsePolicy``,
    ``Agent``, ``OutputContract``, ``OutputContractOption``, ``AgentRequest``), the
    generic ``OPTION_TYPE_DEF`` and ``OPTIONAL_TYPE_DEF``, and
    ``BUILTIN_EXCEPTION_TYPE_DEFS`` (every
    entry of ``semantics.types.BUILTIN_EXCEPTIONS``).
    """
    table = TypeTable()
    for typedef in BUILTIN_PRELUDE_MEMBER_TYPE_DEFS.values():
        table.register(typedef)
    for typedef in BUILTIN_PRELUDE_TYPE_DEFS.values():
        table.register(typedef)
    table.register(OPTION_TYPE_DEF)
    table.register(OPTIONAL_TYPE_DEF)
    for typedef in BUILTIN_EXCEPTION_TYPE_DEFS.values():
        table.register(typedef)
    return table
