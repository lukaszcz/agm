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
receiver constructor. :meth:`TypeTable.method_candidates` returns every
declaration sharing an owner and name as one flat selection level: an
exception's own methods plus its ancestors', and a record's own methods plus
its counted owning enums' (:meth:`TypeTable.owning_enums_for_selection`).

``comparable_types``/``satisfies`` live here rather than in ``semantics.types``
because their record/enum/exception arms consult the table's declaration-level
property flags instead of walking embedded fields; ``semantics.types`` cannot
import this module without a circular import. Those flags are one fixpoint per
:class:`~agm.agl.semantics.analyses.DataProperty`
(``semantics.analyses.compute_declaration_flags``, cycle-safe by
construction), cached on :class:`TypeTable` and invalidated whenever the
table's declarations change. ``EQ`` backs ``=``/``!=`` (``comparable_types``);
``JSON_CONVERTIBLE`` backs :meth:`TypeTable.nominal_is_json_convertible`;
``HASHABLE`` backs the ``Hashable`` constraint; ``EXTERN_KEYABLE`` backs
:func:`is_extern_keyable`. :meth:`TypeTable.nominal_satisfies` takes the
language-level ``ConstraintKind`` (``Eq``/``Hashable``) and maps it onto its
``DataProperty``; the JSON/extern properties have no language-level
constraint spelling, so their table methods use the fixpoint directly.

:func:`satisfies` checks a structural constraint (``Eq``/``Hashable``, see
``agl.constraints``) against a type variable's in-scope bounds, or open-world
mode (``bounds is None``, what :meth:`TypeTable.nominal_reaches_non_data`
uses) where a type variable, the bottom type, and an unresolved inference
variable all count as satisfied. Its nominal case consults the same
declaration-flags fixpoint via :meth:`TypeTable.nominal_satisfies`.
:func:`comparable_types` instead always takes the checker's real bound
environment. :func:`is_json_convertible`/:func:`is_extern_keyable` are
separate structural walks with no bounds concept at all — a bare type
variable is never convertible; the extern key rule instead assumes every
type variable in a key ``Hashable``.

:meth:`TypeTable.has_finite_schema` answers a related but distinct
whole-type question: not "does this type
support ``=``?" but "is this type's reachable *instantiation closure* finite
(so it has a finite JSON schema)?" — a generic recursive declaration may
reference itself at ever-larger arguments (polymorphic recursion), which
never blocks construction/matching/equality but does mean no finite schema
exists. Backed by ``semantics.analyses.compute_finite_closure``, cached and
invalidated the same way as the declaration-flags fixpoints above.
:meth:`TypeTable.first_infinite_declaration`/:meth:`TypeTable.no_finite_schema_message`
build on the same query to name the culprit declaration for a use-site
diagnostic (agent output target, cast target, parameter type).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, assert_never, cast

from agm.agl.constraints import ConstraintBounds, ConstraintKind
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import (
    NO_DECL_ID,
    require_reserved_enum_member_id,
)
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id as _reserved_id
from agm.agl.modules.ids import RESERVED_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.external_names import NO_EXTERNAL_NAME, ExternalName
from agm.agl.semantics.types import (
    EXCEPTION_BASE,
    HOST_MINTED_PRELUDE_TYPE_IDS,
    HOST_MINTED_PRELUDE_TYPE_NAMES,
    ArrayType,
    BoolType,
    BottomType,
    CastKind,
    CheckedType,
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
    contains_inference_var,
    contains_type_var,
    free_type_vars,
    is_assignable,
    is_scalar_json_shaped,
    match_nominal_owner_template,
    match_type_template,
    spells_bare,
    standard_option_type,
    standard_optional_type,
    substitute,
    type_children,
)
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue, Value
from agm.agl.zones import ParamZone
from agm.util.graph import bfs_first

if TYPE_CHECKING:
    from agm.agl.semantics.analyses import DeclarationFlags, FiniteClosure

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
    many leading method type parameters belong to the receiver type. Its
    constraint block lives on the method's own ``FunctionSignature``
    (``TypeEnvironment.get_function_signature_by_node_id(decl_node_id)``), not
    here.
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


def method_receiver_match(method: MethodDef, owner: Type) -> TypeTemplateMatch | None:
    """Match a built-in method's receiver template against a concrete *owner*, for SELECTION.

    Only the method's own receiver-prefix type parameters
    (``method.type_params[:method.receiver_type_param_arity]``) are free in
    the receiver template: a concrete ``dict[text, V]`` receiver (no free key
    parameter) matches only a text-keyed dict, while a bare ``dict[K, V]``
    receiver matches any dict, binding both ``K`` and ``V``. Used by
    :meth:`TypeTable.method_candidates` to decide candidacy; an owner
    position still uninferred (e.g. an empty ``{}`` literal's key/value) is
    treated as a wildcard (``wildcard_inference_vars``) so a candidate is not
    ruled out before inference has run. Specialization
    (``typecheck.checker._Checker._bound_method_type``) does NOT reuse this
    match's bindings — it re-derives them through real unification, since a
    wildcard match carries no real binding.
    """
    receiver_type_params = method.type_params[: method.receiver_type_param_arity]
    return match_type_template(
        method.signature.params[0], owner, receiver_type_params, wildcard_inference_vars=True
    )


class DataProperty(enum.Enum):
    """One declaration-level structural property computed by the shared fixpoint.

    ``EQ``/``HASHABLE`` back the language-level ``Eq``/``Hashable``
    constraints (``agm.agl.constraints.ConstraintKind``); ``JSON_CONVERTIBLE``
    backs ``as json``/``as text`` and every wire boundary; ``EXTERN_KEYABLE``
    backs extern signatures. Each has its own :class:`LeafPolicy` in
    :data:`LEAF_POLICIES`. The shared fixpoint itself
    (:func:`~agm.agl.semantics.analyses.compute_declaration_flags`) lives in
    ``semantics.analyses``, which imports this enum and :data:`LEAF_POLICIES`
    from here.
    """

    EQ = "eq"
    HASHABLE = "hashable"
    JSON_CONVERTIBLE = "json_convertible"
    EXTERN_KEYABLE = "extern_keyable"


@dataclass(frozen=True, slots=True)
class LeafPolicy:
    """What counts as "bad" evidence for one declaration-level fixpoint.

    ``recurse_containers`` — ``True`` to recurse into an ``array``/``dict``'s
    element/value type; ``False`` to flag the container outright.
    ``var_fields_bad`` — whether a declaration's own ``var`` field (or, for
    an enum, one of its members' own) is itself bad.
    ``non_data_bad`` — whether a function or ``unit`` leaf is bad outright;
    when ``False``, a function leaf is instead evaluated by recursing into
    its parameter and result types (``unit`` is then never bad).
    ``dict_key_ok`` — ``None`` when a ``dict``'s key type is never itself bad
    (its own recursion covers key and value alike); otherwise a per-policy
    key-form check over a key position (a direct ``dict`` key, or a type
    argument filling another declaration's own key parameter): the key is bad
    outright unless the check accepts it (still subject to
    ``recurse_containers`` for the value type either way). Its third
    argument is the set of the enclosing declaration's own type parameters
    to assume satisfy the rule — deferred, like any bare type variable, to
    whatever concrete argument a later reference supplies for them; empty at
    a closed, concrete reference site.
    """

    recurse_containers: bool
    var_fields_bad: bool
    non_data_bad: bool
    dict_key_ok: Callable[[Type, "TypeTable", frozenset[str]], bool] | None


#: ``Hashable`` with its implication (``Eq``) already applied.
_HASHABLE_CLOSED = frozenset({ConstraintKind.HASHABLE, ConstraintKind.EQ})
_NO_BOUNDS: ConstraintBounds = MappingProxyType({})


def dict_key_is_hashable(
    key: Type, table: "TypeTable", assume_ok: frozenset[str] = frozenset()
) -> bool:
    """``JSON_CONVERTIBLE``'s dict-key rule: the key must be ``Hashable``.

    *assume_ok* — see :attr:`LeafPolicy.dict_key_ok`.
    """
    bounds = {p: _HASHABLE_CLOSED for p in assume_ok} if assume_ok else _NO_BOUNDS
    return satisfies(key, ConstraintKind.HASHABLE, table, bounds)


def dict_key_is_hashable_assuming_type_vars(
    key: Type, table: "TypeTable", assume_ok: frozenset[str] = frozenset()
) -> bool:
    """``EXTERN_KEYABLE``'s dict-key rule: ``Hashable``, assuming every free type variable is.

    Extern code builds keys of a type variable's instantiation, so a type
    variable (bounded or not) passes wherever it occurs in the key.
    *assume_ok* — see :attr:`LeafPolicy.dict_key_ok`.
    """
    return dict_key_is_hashable(key, table, assume_ok | free_type_vars(key))


#: Property -> the policy computing its declaration-level "does not satisfy"
#: set (see ``semantics.analyses.compute_declaration_flags``): EQ flags a
#: declaration that unconditionally reaches ``unit``/a function type,
#: recursing into ``array``/``dict``; HASHABLE flags one that is not deeply
#: immutable data — a function/unit/``var`` field, or an ``array``/``dict``
#: outright (never recursed into); JSON_CONVERTIBLE additionally flags a dict
#: keyed by a non-``Hashable`` type; EXTERN_KEYABLE instead requires a key
#: that is ``Hashable`` assuming its type variables are and, unlike
#: JSON_CONVERTIBLE, a function leaf is not bad (a callback's parameters are
#: built by the companion), so it recurses into function types instead of
#: flagging them.
LEAF_POLICIES: Mapping[DataProperty, LeafPolicy] = MappingProxyType(
    {
        DataProperty.EQ: LeafPolicy(
            recurse_containers=True,
            var_fields_bad=False,
            non_data_bad=True,
            dict_key_ok=None,
        ),
        DataProperty.HASHABLE: LeafPolicy(
            recurse_containers=False,
            var_fields_bad=True,
            non_data_bad=True,
            dict_key_ok=None,
        ),
        DataProperty.JSON_CONVERTIBLE: LeafPolicy(
            recurse_containers=True,
            var_fields_bad=False,
            non_data_bad=True,
            dict_key_ok=dict_key_is_hashable,
        ),
        DataProperty.EXTERN_KEYABLE: LeafPolicy(
            recurse_containers=True,
            var_fields_bad=False,
            non_data_bad=False,
            dict_key_ok=dict_key_is_hashable_assuming_type_vars,
        ),
    }
)


_KIND_PROPERTIES: Mapping[ConstraintKind, DataProperty] = MappingProxyType(
    {ConstraintKind.EQ: DataProperty.EQ, ConstraintKind.HASHABLE: DataProperty.HASHABLE}
)


def _constraint_kind_to_data_property(kind: ConstraintKind) -> DataProperty:
    """Map a language-level constraint kind onto its declaration-flags property.

    The single place ``ConstraintKind`` (``Eq``/``Hashable``) is translated to
    a :class:`DataProperty`; ``JSON_CONVERTIBLE`` and ``EXTERN_KEYABLE``
    have no language-level constraint spelling.
    """
    return _KIND_PROPERTIES[kind]


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
                   chain).
    ``field_has_default`` — whether each field has a declared default,
                   strictly paired with ``fields`` like ``field_kinds``.
                   Construction may leave it ``None``, which
                   ``__post_init__`` normalizes to an all-``False`` tuple the
                   length of ``fields`` — the common case for a declaration
                   with no defaulted fields, sparing every such call site an
                   explicit all-``False`` literal; a *typedef* is therefore
                   never actually seen holding ``None`` once built. The
                   default EXPRESSION itself is never stored here — presence
                   is a fact about the type's shape, but the expression is
                   ordinary code, so it reaches later passes the same route a
                   function parameter default does: lowered with the
                   declaration into the constructor descriptor's own
                   ``IrFunctionParam.default``-shaped slot, independent of
                   ``syntax``/``ir``. For an exception, its OWN presence
                   flags only (see :meth:`TypeTable.field_has_default` for
                   the flattened base chain, which inherits a base field's
                   default unchanged).
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
                   created by an inline enum member, as opposed to a
                   separately declared record an enum references.
    ``external_name`` — a record's own ``@name``/``@json-name`` spellings
                   (its value-syntax name and its JSON tag as an enum
                   member); unused for enums and exceptions.
    ``field_external_names`` — ``(field_name, ExternalName)`` pairs for the
                   OWN fields carrying ``@name``/``@json-name``, in
                   declaration order; unrenamed fields are absent.
    ``doc``        — the declaration's recognized ``@doc`` prose, when present.
                   It is presentation metadata rather than part of the type's
                   semantic shape, so it is excluded from equality/hashing.
    ``field_docs`` — ``(field_name, doc)`` pairs for the OWN fields carrying
                   ``@doc``, in declaration order; excluded from
                   equality/hashing like ``doc``.
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
    field_has_default: tuple[bool, ...] | None = None
    is_builtin: bool = field(default=False, compare=False)
    decl_node_id: int = field(default=NO_DECL_ID, compare=False)
    is_inline_enum_member: bool = field(default=False, compare=False)
    external_name: ExternalName = NO_EXTERNAL_NAME
    field_external_names: tuple[tuple[str, ExternalName], ...] = ()
    doc: str | None = field(default=None, compare=False)
    field_docs: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Normalize an omitted ``field_has_default`` into all-``False``.

        Every construction site that declares no defaulted field may simply
        leave ``field_has_default`` unset; this fills the length ``fields``
        requires so a *typedef* is never actually seen holding ``None``.
        Normalizing here (not lazily in an accessor) is also what keeps
        equality meaningful: a seeded canonical literal that omits the
        argument and a source declaration whose builder computed an explicit
        all-``False`` tuple end up holding the identical value.
        """
        if self.field_has_default is None:
            object.__setattr__(self, "field_has_default", (False,) * len(self.fields))

    def handle(self, type_args: tuple[Type, ...] = ()) -> RecordType | EnumType | ExceptionType:
        """Return the ``RecordType``/``EnumType``/``ExceptionType`` handle naming this ``TypeDef``.

        Convenience for call sites that hold a ``TypeDef`` and need the
        corresponding handle (e.g. to register a value, or to pass to
        :meth:`TypeTable.record_fields`/:meth:`TypeTable.enum_members`/
        :meth:`TypeTable.exception_fields`). *type_args* defaults to ``()``
        for non-generic defs and is empty for an exception (exceptions are
        never generic). The returned handle's ``decl_id`` is stamped from
        ``self.decl_node_id``.
        """
        match self.kind:
            case "record":
                return self.record_handle(type_args)
            case "enum":
                return self.enum_handle(type_args)
            case "exception":
                return self.exception_handle()
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def record_handle(self, type_args: tuple[Type, ...] = ()) -> RecordType:
        """Return the ``RecordType`` handle naming this record ``TypeDef``."""
        return RecordType(
            name=self.name,
            type_args=type_args,
            module_id=self.module_id,
            scope_path=self.scope_path,
            decl_id=self.decl_node_id,
        )

    def enum_handle(self, type_args: tuple[Type, ...] = ()) -> EnumType:
        """Return the ``EnumType`` handle naming this enum ``TypeDef``."""
        return EnumType(
            name=self.name,
            type_args=type_args,
            module_id=self.module_id,
            scope_path=self.scope_path,
            decl_id=self.decl_node_id,
        )

    def exception_handle(self) -> ExceptionType:
        """Return the ``ExceptionType`` handle naming this exception ``TypeDef``."""
        return ExceptionType(
            name=self.name,
            module_id=self.module_id,
            scope_path=self.scope_path,
            decl_id=self.decl_node_id,
        )


@dataclass(frozen=True, slots=True)
class NonDataLeaf:
    """A non-data type (``unit`` or function) reached directly in a type's own structure."""

    leaf: Type


@dataclass(frozen=True, slots=True)
class NonDataField:
    """A declaration field whose own type structurally reaches a non-data leaf."""

    typedef: TypeDef
    field_name: str
    field_type: Type


@dataclass(frozen=True, slots=True)
class BadDictKey:
    """A non-``Hashable`` dict key type reached directly in a type's own structure."""

    key_type: Type


@dataclass(frozen=True, slots=True)
class BadDictKeyField:
    """A declaration field whose own dict key type is directly not ``Hashable``."""

    typedef: TypeDef
    field_name: str
    key_type: Type


@dataclass(frozen=True, slots=True)
class BadKeyArgument:
    """A reference supplies *argument* for *target*'s own dict-key parameter *param_name*, and
    *argument* fails the key rule.

    *source*/*field_name* name the field of *source* whose type carries the
    offending reference, or are both ``None`` when the reference is the
    examined type's own handle (a generic reference filling its own
    declaration's key parameter directly, with no enclosing field).
    """

    source: TypeDef | None
    field_name: str | None
    target: TypeDef
    param_name: str
    argument: Type


@dataclass(frozen=True, slots=True)
class FreeTypeVar:
    """A free type variable that may stand for a type with no JSON representation."""

    name: str


#: Why a type has no JSON representation, as one tagged structural culprit — see
#: :meth:`TypeTable.json_representation_culprit`.
JsonRepresentationCulprit = (
    NonDataLeaf | NonDataField | BadDictKey | BadDictKeyField | BadKeyArgument | FreeTypeVar
)


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
        # Memo for field_has_default's exception branch — same keying convention.
        self._exception_field_has_default_cache: dict[DeclId, tuple[tuple[str, bool], ...]] = {}
        # Bare name -> the handle of a standard-library ``builtin exception``,
        # published by its shell registration (see
        # :meth:`declare_standard_builtin_exception`). Not a cache of ``_defs``
        # and never invalidated from it: it records identities, which are known
        # a whole phase earlier than the definitions they name. Scoped to the
        # compile that publishes it and deliberately not carried by
        # :meth:`merge_from`, so a table seeded from another one starts empty
        # here; :meth:`exception_root`'s fallback over ``_defs`` answers with
        # the same identity until the next check re-runs the shells.
        self._standard_builtin_exceptions: dict[str, ExceptionType] = {}
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
        # Declaration key -> the one candidate map holding it. Registering a
        # method first retires its declaration wherever it stood, so each key
        # occupies exactly one map and supersession is a lookup rather than a
        # walk of every receiver's methods.
        self._method_sites: dict[DeclKey, dict[DeclKey, MethodDef]] = {}
        # Whole-table declaration-flags fixpoint (see :meth:`nominal_satisfies`),
        # one result per DataProperty, computed lazily on first use and
        # invalidated (cleared) whenever a declaration is added, removed, or
        # overwritten.
        self._declaration_flags_cache: dict[DataProperty, DeclarationFlags] = {}
        # Concrete nominal types whose successful Hashable checks may be used
        # by retained REPL code. Merged with declaration state across entries.
        self._hashable_proofs: set[RecordType | EnumType | ExceptionType] = set()
        # Member declaration id -> the enums declaring or referencing it, in
        # registration order. A referenced member belongs to several enums, so
        # this is multi-valued. Whole-table, rebuilt on any registration change.
        self._member_enum_owners: dict[DeclId, tuple[DeclId, ...]] | None = None
        # Whole-table finiteness fixpoint (see :meth:`has_finite_schema`),
        # cached and invalidated the same way as ``_declaration_flags_cache``.
        self._finite_closure: FiniteClosure | None = None
        # Whole-table relevant type parameters, shared by both fixpoints.
        self._relevant_params: Mapping[DeclId, frozenset[str]] | None = None
        # Exception declaration id -> its direct children's ids, built in one
        # pass over ``_defs``. Whole-table, rebuilt on any registration
        # change; feeds :meth:`exception_descendants`.
        self._exception_children: dict[DeclId, tuple[DeclId, ...]] | None = None
        # Enum identity -> its inline members by terminal name, each as the
        # enum's type parameters and the member's handle template over
        # them. Declaration shells declare
        # them a phase before the enum's body registers (see
        # :meth:`declare_inline_member`); a superseded enum keeps its own.
        self._inline_members: dict[DeclId, dict[str, tuple[tuple[str, ...], RecordType]]] = {}

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

        When self-validation is enabled, also rejects a *typedef* whose
        ``field_has_default`` does not have one entry per ``fields`` entry —
        the same positional-pairing invariant :meth:`field_kinds` relies on,
        but caught immediately here rather than as a distant ``zip(...,
        strict=True)`` failure the first time some unrelated caller reads
        :meth:`field_has_default` for it.
        """
        if self_validation_enabled() and typedef.decl_node_id == NO_DECL_ID:
            raise AssertionError(
                f"cannot register a TypeDef with no declaration identity: {typedef!r}"
            )
        if self_validation_enabled() and len(self._own_field_has_default(typedef)) != len(
            typedef.fields
        ):
            raise AssertionError(
                f"TypeDef {typedef.name!r} declares {len(typedef.fields)} field(s) but "
                f"field_has_default has {len(typedef.field_has_default or ())} entries: "
                f"{typedef!r}"
            )
        decl_id = typedef.decl_node_id
        existing = self._defs.get(decl_id)
        self._name_index[(typedef.module_id, typedef.scope_path, typedef.name)] = decl_id
        if existing is None:
            self._defs[decl_id] = typedef
            if typedef.kind == "enum":
                member_path = (*typedef.scope_path, typedef.name)
                for member in typedef.members:
                    if member.module_id == typedef.module_id and member.scope_path == member_path:
                        self.declare_inline_member(decl_id, typedef.type_params, member)
            self._standard_builtins = None
            self._host_minted_ids = None
            self._declaration_flags_cache = {}
            self._member_enum_owners = None
            self._finite_closure = None
            self._relevant_params = None
            self._exception_children = None
            return
        if self_validation_enabled() and existing != typedef:
            raise AssertionError(
                f"conflicting TypeDef registration for identity {decl_id!r}: "
                f"{existing!r} is already registered, got {typedef!r}"
            )

    def declare_inline_member(
        self, enum_id: DeclId, type_params: tuple[str, ...], member: RecordType
    ) -> None:
        """Record *member*'s handle template as an inline member of enum *enum_id*.

        *member*'s type arguments range over the enum's *type_params*. The
        builder declares each member from its shell, before the enum's body
        registers, so :meth:`inline_member` answers while bodies are resolved.
        """
        self._inline_members.setdefault(enum_id, {})[member.name] = (type_params, member)

    def inline_member(self, owner: EnumType, name: str) -> RecordType:
        """Return *owner*'s inline member *name* at *owner*'s type arguments.

        Scope decides owner-member selection for every position and rejects
        any spelling naming no such member before this is ever called, so
        *name* always names a declared member of *owner* here.
        """
        type_params, member = self._inline_members[owner.decl_id][name]
        return substitute(member, dict(zip(type_params, owner.type_args, strict=True)))

    def get(
        self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()
    ) -> TypeDef | None:
        """Return the newest registered ``TypeDef`` for a name path, via the name index."""
        decl_id = self._name_index.get((module_id, scope_path, name))
        return None if decl_id is None else self._defs[decl_id]

    def named(self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()) -> TypeDef:
        """Return the newest ``TypeDef`` registered for a name path a checked program declares."""
        return self._defs[self._name_index[(module_id, scope_path, name)]]

    def typedef_of(self, decl_id: DeclId) -> TypeDef:
        """Return the ``TypeDef`` registered for a declaration a checked program declares."""
        return self._defs[decl_id]

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
        :meth:`~agm.agl.typecheck.env.TypeEnvironment.rewind_from`).
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
        """Remove a superseded method declaration from the receiver holding it."""
        site = self._method_sites.pop(declaration_key, None)
        if site is not None:
            del site[declaration_key]

    def _forget_methods_of(self, decl_id: DeclId) -> None:
        """Drop every method a superseded declaration owns, index included."""
        for candidates in self._methods.pop(decl_id, {}).values():
            for key in candidates:
                del self._method_sites[key]

    def _file_method(self, candidates: dict[DeclKey, MethodDef], method: MethodDef) -> None:
        """Move *method*'s declaration into *candidates*, recording where it lands."""
        self._remove_method_declaration(method.declaration_key)
        candidates[method.declaration_key] = method
        self._method_sites[method.declaration_key] = candidates

    def _put_method(self, decl_id: DeclId, method: MethodDef) -> None:
        """Register *method* under its declaration key on *decl_id*."""
        self._file_method(self._methods.setdefault(decl_id, {}).setdefault(method.name, {}), method)

    def register_method(self, owner: NominalOwner, method: MethodDef) -> None:
        """Register *method* under its nominal *owner*."""
        self._put_method(owner.decl_id, method)

    def register_builtin_method(self, constructor: str, method: MethodDef) -> None:
        """Register *method* under a built-in receiver type constructor."""
        self._file_method(
            self._builtin_methods.setdefault(constructor, {}).setdefault(method.name, {}), method
        )

    def rewind_methods_from(self, previous: TypeTable, declaration_ids: Collection[int]) -> None:
        """Undo the method registrations *declaration_ids* made, restoring *previous*'s.

        The method-table half of
        :meth:`~agm.agl.typecheck.env.TypeEnvironment.rewind_from`, which is
        the only caller: methods are keyed by declaration identity here, so
        the declarations an entry did not promote name their own
        registrations directly.
        """
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
        for key in declaration_keys:
            self._remove_method_declaration(key)
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

    def method_candidates(self, owner: NominalOwner | Type, name: str) -> tuple[MethodDef, ...]:
        """Return *owner*'s candidates for *name* as one flat selection level.

        A record's level is its own methods plus those of every counted
        owning enum (:meth:`owning_enums_for_selection`); an exception's
        level is its own plus every ancestor's (:meth:`ancestor_defs`,
        nearest first); an enum's level is its own methods only. Builtin
        receivers are flat by construction. No candidate has priority over
        another within the level — selection visibility and ambiguity are a
        caller concern.
        """
        constructor = self._builtin_constructor(owner)
        if constructor is not None:
            level = self._builtin_methods.get(constructor, {}).get(name, {})
            candidates = self._method_level(level)
            return tuple(
                candidate
                for candidate in candidates
                if method_receiver_match(candidate, owner) is not None
            )
        if not isinstance(owner, (RecordType, EnumType, ExceptionType)):
            return ()
        owner_ids: tuple[DeclId, ...] = (owner.decl_id,)
        if isinstance(owner, ExceptionType):
            owner_ids += tuple(base.decl_node_id for base in self.ancestor_defs(owner.decl_id))
        elif isinstance(owner, RecordType):
            owner_ids += tuple(
                enum_def.decl_node_id
                for enum_def, _bindings in self.owning_enums_for_selection(owner)
            )
        return tuple(
            method
            for decl_id in owner_ids
            for method in self._method_level(self._methods.get(decl_id, {}).get(name, {}))
        )

    def declared_methods(self, owner_id: DeclId) -> Mapping[str, tuple[MethodDef, ...]]:
        """Return every method directly declared for each name on *owner_id*."""
        return {
            name: self._method_level(methods)
            for name, methods in self._methods.get(owner_id, {}).items()
            if methods
        }

    def _exception_chain(self, decl_id: DeclId) -> list[tuple[DeclId, TypeDef]]:
        """Return *decl_id*'s base chain, base first.

        Every flattened exception accessor inherits base-first declaration
        order from this one walk. The whole-program inhabitation fixpoint
        rejects a cyclic ``extends`` chain before any exception's fields are
        ever flattened, so the walk here is unconditionally finite.
        """
        chain: list[tuple[DeclId, TypeDef]] = []
        current: DeclId | None = decl_id
        while current is not None:
            typedef = self._defs[current]
            chain.append((current, typedef))
            current = typedef.base
        chain.reverse()
        return chain

    def exception_chain_defs(self, decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return *decl_id*'s exception base chain, base first, own def last.

        The shared walk every flattened exception accessor
        (:meth:`exception_fields`, :meth:`field_kinds`, :meth:`field_has_default`,
        :meth:`field_external_names`) builds on, exposed for a caller outside
        this module that needs the same base-first order over its own
        per-declaration data (e.g. a lowered field default keyed by
        declaration identity) instead of reimplementing the chain walk.
        """
        return tuple(typedef for _decl_id, typedef in self._exception_chain(decl_id))

    def ancestor_defs(self, decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return *decl_id*'s exception ancestors, nearest first.

        Empty for a hierarchy root and for any non-exception declaration, so a
        caller checking inherited members needs no base-chain walk of its own.
        """
        typedef = self._defs[decl_id]
        if typedef.kind != "exception" or typedef.base is None:
            return ()
        chain = self._exception_chain(typedef.base)
        return tuple(base_def for _base_id, base_def in reversed(chain))

    def is_exception_ancestor(self, ancestor_id: DeclId, decl_id: DeclId) -> bool:
        """Whether *ancestor_id* is in *decl_id*'s exception base chain."""
        return any(ancestor.decl_node_id == ancestor_id for ancestor in self.ancestor_defs(decl_id))

    def _exception_children_index(self) -> Mapping[DeclId, tuple[DeclId, ...]]:
        """Return the memoized exception declaration id -> direct children ids index.

        Built in one pass over every registered exception, skipping an
        orphaned one on either end of the parent-child link (see
        :meth:`orphan`).
        """
        index = self._exception_children
        if index is None:
            built: dict[DeclId, list[DeclId]] = {}
            for decl_id, typedef in self._defs.items():
                if typedef.kind != "exception" or decl_id in self._orphaned:
                    continue
                if typedef.base is not None and typedef.base not in self._orphaned:
                    built.setdefault(typedef.base, []).append(decl_id)
            index = {parent: tuple(children) for parent, children in built.items()}
            self._exception_children = index
        return index

    def exception_children(self, decl_id: DeclId) -> tuple[DeclId, ...]:
        """Return the ids of the exceptions directly extending *decl_id*."""
        return self._exception_children_index().get(decl_id, ())

    def exception_descendants(self, decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return every exception descending from *decl_id*, breadth-first.

        Reads the memoized parent-to-children index, so a caller checking one
        owner never rescans the whole table.
        """
        index = self._exception_children_index()
        result: list[TypeDef] = []
        frontier = [decl_id]
        while frontier:
            next_frontier: list[DeclId] = []
            for parent in frontier:
                for child_id in index.get(parent, ()):
                    result.append(self._defs[child_id])
                    next_frontier.append(child_id)
            frontier = next_frontier
        return tuple(result)

    def is_builtin_exception_root(self, decl_id: DeclId) -> bool:
        """Whether *decl_id* is the built-in ``Exception``, reserved or loaded.

        It is the only exception without a base: an omitted ``extends`` means
        ``extends Exception``.
        """
        typedef = self._defs.get(decl_id)
        return typedef is not None and typedef.kind == "exception" and typedef.base is None

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
        self._exception_field_has_default_cache.clear()
        # The declaration-flags and finiteness fixpoints are whole-table (any
        # declaration's flag can in principle depend on any other's), so a
        # single changed identity invalidates the whole cached result rather
        # than just this one.
        self._standard_builtins = None
        self._host_minted_ids = None
        self._declaration_flags_cache = {}
        self._member_enum_owners = None
        self._finite_closure = None
        self._relevant_params = None
        self._exception_children = None

    def record_fields(self, handle: RecordType) -> Mapping[str, Type]:
        """Return *handle*'s field types with its ``type_args`` substituted in.

        Memoized per handle: ``RecordType`` equality/hash cover ``decl_id``
        and ``type_args``, so the same handle always maps to the same
        substituted mapping object. The memo is bucketed by ``decl_id`` so a single
        identity's invalidation (:meth:`merge_from`) never has to scan entries
        for other identities.
        """
        decl_id = handle.decl_id
        bucket = self._record_fields_cache.get(decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._defs[decl_id]
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
        """
        return self._defs[handle.decl_id].mutable_fields

    def enum_members(self, handle: EnumType) -> tuple[RecordType, ...]:
        """Return *handle*'s member record types with ``type_args`` substituted in.

        Members retain their declaration identities and field ownership. The
        result is memoized per enum instantiation.
        """
        decl_id = handle.decl_id
        bucket = self._enum_members_cache.get(decl_id)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._defs[decl_id]
        subst = dict(zip(typedef.type_params, handle.type_args))
        result = tuple(substitute(member, subst) for member in typedef.members)
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

    def _enum_defs_owning_id(self, record_decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return the enum definitions naming *record_decl_id* a member, registration order."""
        owners = self._member_enum_owner_index().get(record_decl_id, ())
        return tuple(self._defs[owner_id] for owner_id in owners)

    def _enum_defs_owning(self, record: RecordType) -> tuple[TypeDef, ...]:
        """Return the enum definitions *record* is a member of, in registration order."""
        return self._enum_defs_owning_id(record.decl_id)

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

    def _enum_membership_matches(
        self, owner_defs: tuple[TypeDef, ...], record: RecordType
    ) -> Iterator[tuple[TypeDef, TypeTemplateMatch]]:
        """Yield each of *owner_defs* paired with its member match for *record*.

        Skips an owner whose member template does not bind against *record*.
        """
        for typedef in owner_defs:
            member = next(item for item in typedef.members if item.decl_id == record.decl_id)
            match = self._match_enum_member_template(typedef, member, record)
            if match is not None:
                yield typedef, match

    def owning_enum_defs_for_selection(self, record_decl_id: DeclId) -> tuple[TypeDef, ...]:
        """Return the record identity's counted owning enum defs, registration order.

        Every *current* enum that declares or references the record; a
        superseded owning enum is skipped unless the record itself is
        superseded, so a retained member of a superseded enum keeps seeing
        that enum's methods.
        """
        record_def = self._defs.get(record_decl_id)
        record_is_current = record_def is not None and self.is_current(record_def)
        return tuple(
            enum_def
            for enum_def in self._enum_defs_owning_id(record_decl_id)
            if not (record_is_current and not self.is_current(enum_def))
        )

    def owning_enums_for_selection(
        self, record: RecordType
    ) -> tuple[tuple[TypeDef, Mapping[str, Type]], ...]:
        """Return *record*'s counted owning enums, with their partial bindings.

        See :meth:`owning_enum_defs_for_selection` for which enums count. A
        captured enum parameter's binding is present; a phantom (uncaptured)
        parameter is absent.
        """
        owner_defs = self.owning_enum_defs_for_selection(record.decl_id)
        return tuple(
            (enum_def, dict(match.bindings))
            for enum_def, match in self._enum_membership_matches(owner_defs, record)
        )

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

    def record_matches_enum_member(
        self, enum: EnumType, type_params: tuple[str, ...], member_name: str, record: RecordType
    ) -> bool:
        """Return whether *record* is the inline member *member_name* of *enum* over *type_params*.

        This is what the owner-qualified spelling ``Enum::member_name`` selects.
        The owner's own *type_params* are inferred; every other owner argument
        must match exactly. A name mismatch is rejected directly, without
        consulting :meth:`inline_member`: this compares a matched pattern's
        declared member name against a checked subject's own record type,
        which may be an unrelated record sharing no member with *enum*.
        """
        if record.name != member_name:
            return False
        member = self.inline_member(enum, member_name)
        return (
            member.decl_id == record.decl_id
            and match_nominal_owner_template(TypeTemplate(member, type_params), record) is not None
        )

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
        for _chain_id, typedef in self._exception_chain(decl_id):
            fields.update(typedef.fields)
        return fields

    def field_kinds(self, handle: RecordType | ExceptionType) -> tuple[tuple[str, ParamZone], ...]:
        """Return *handle*'s ``(field_name, ParamZone)`` pairs, in field order.

        For a record, reads the zones straight off its own ``TypeDef``
        (:attr:`TypeDef.field_kinds`) — declaration-level, like
        :meth:`record_mutable_fields`, so every instantiation of a generic
        record shares the same zones regardless of ``type_args`` and no
        memoization is needed. ``field_kinds`` has one entry per field, in
        order.

        For an exception, mirrors :meth:`exception_fields`'s base-chain
        flattening (base fields first, in declaration order, then the
        exception's own), memoized the same way — an exception's OWN fields
        honor their declared ``@arg-*`` attribute exactly like a record's
        fields do; only inheritance is exception-specific.
        """
        if isinstance(handle, ExceptionType):
            decl_id = handle.decl_id
            cached = self._exception_field_kinds_cache.get(decl_id)
            if cached is not None:
                return cached
            result = self._flatten_exception_field_kinds(decl_id)
            self._exception_field_kinds_cache[decl_id] = result
            return result
        typedef = self._defs[handle.decl_id]
        return tuple(
            zip((fname for fname, _ftype in typedef.fields), typedef.field_kinds, strict=True)
        )

    def _flatten_exception_field_kinds(self, decl_id: DeclId) -> tuple[tuple[str, ParamZone], ...]:
        return tuple(
            (fname, kind)
            for _chain_id, typedef in self._exception_chain(decl_id)
            for (fname, _ftype), kind in zip(typedef.fields, typedef.field_kinds, strict=True)
        )

    @staticmethod
    def _own_field_has_default(typedef: TypeDef) -> tuple[bool, ...]:
        """Return *typedef*'s own per-field default-presence tuple.

        Always a concrete tuple: :meth:`TypeDef.__post_init__` normalizes an
        omitted (``None``) construction argument into all-``False`` of
        ``fields``' length before any ``TypeDef`` instance is observable.
        """
        assert typedef.field_has_default is not None
        return typedef.field_has_default

    def field_has_default(self, handle: RecordType | ExceptionType) -> tuple[tuple[str, bool], ...]:
        """Return *handle*'s ``(field_name, has_default)`` pairs, in field order.

        Whether each field carries a declared default (see
        :attr:`TypeDef.field_has_default`) — the default EXPRESSION itself is
        not carried here; it reaches lowering the same route a function
        parameter default does (see :attr:`TypeDef.field_has_default`'s
        docstring). Mirrors :meth:`field_kinds` exactly: a record reads
        straight off its own ``TypeDef`` (declaration-level, so every
        instantiation of a generic record shares the same defaults); an
        exception flattens the ``extends`` base chain, base fields first, so
        an inherited field keeps its base's default presence.

        Consumes declarations already checked for acyclic inheritance.
        """
        if isinstance(handle, ExceptionType):
            decl_id = handle.decl_id
            cached = self._exception_field_has_default_cache.get(decl_id)
            if cached is not None:
                return cached
            result = self._flatten_exception_field_has_default(decl_id)
            self._exception_field_has_default_cache[decl_id] = result
            return result
        typedef = self._defs[handle.decl_id]
        return tuple(
            zip(
                (fname for fname, _ftype in typedef.fields),
                self._own_field_has_default(typedef),
                strict=True,
            )
        )

    def _flatten_exception_field_has_default(self, decl_id: DeclId) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (fname, has_default)
            for _chain_id, typedef in self._exception_chain(decl_id)
            for (fname, _ftype), has_default in zip(
                typedef.fields, self._own_field_has_default(typedef), strict=True
            )
        )

    def field_external_names(
        self, handle: RecordType | ExceptionType
    ) -> Mapping[str, ExternalName]:
        """Return the renamed fields of *handle*, an exception's base chain applied.

        A field without ``@name``/``@json-name`` is absent from the mapping.
        """
        if isinstance(handle, RecordType):
            return dict(self._defs[handle.decl_id].field_external_names)
        return {
            field_name: external
            for _chain_id, typedef in self._exception_chain(handle.decl_id)
            for field_name, external in typedef.field_external_names
        }

    def external_name(self, handle: RecordType) -> ExternalName:
        """Return the ``@name``/``@json-name`` spellings of record *handle*'s declaration."""
        return self._defs[handle.decl_id].external_name

    def member_json_tag(self, member: RecordType, name: str) -> str:
        """An enum member's effective JSON ``$case`` tag (``@json-name`` ?? ``@name`` ?? *name*)."""
        return self.external_name(member).json(name)

    def declaration_doc(self, handle: RecordType | EnumType) -> str | None:
        """Return a record, member, or enum declaration's recognized ``@doc`` prose."""
        return self._defs[handle.decl_id].doc

    def field_docs(self, handle: RecordType) -> Mapping[str, str]:
        """Return the ``@doc`` prose of record *handle*'s documented fields."""
        return dict(self._defs[handle.decl_id].field_docs)

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
        which lives here rather than on the ``ExceptionType`` handle.
        """
        return self._defs[handle.decl_id]

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

    def declare_standard_builtin_exception(self, handle: ExceptionType) -> None:
        """Publish a standard-library ``builtin exception``'s identity.

        Registering a shell establishes a declaration's identity one whole
        phase before its fields — and therefore its definition — can be
        resolved. An exception is never generic, so its handle *is* that
        identity in full, and a caller that needs nothing more
        (:meth:`exception_root`) can be answered from here while bodies are
        still resolving in any order. Definitions are unaffected: the
        ``TypeDef`` this names is registered later, as usual.
        """
        self._standard_builtin_exceptions[handle.name] = handle

    def exception_root(self) -> ExceptionType:
        """Return the built-in ``Exception``: loaded from the standard library, else reserved.

        Prefers the identity published for the declaration being compiled
        here, so an exception that omits ``extends`` names the same root
        whether or not the root's own body has been resolved yet. A program
        whose standard library was restored rather than read — its shells
        never ran here — answers from the registered declaration instead, and
        one loaded without a standard library falls back to the reserved
        identity.
        """
        published = self._standard_builtin_exceptions.get("Exception")
        if published is not None:
            return published
        standard = self.standard_builtin_declaration("Exception")
        return EXCEPTION_BASE if standard is None else standard.exception_handle()

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
        return declaration.enum_handle((argument,))

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

    def nominal_satisfies(
        self,
        handle: RecordType | EnumType | ExceptionType,
        kind: ConstraintKind,
        bounds: ConstraintBounds | None,
    ) -> bool:
        """Return ``True`` if *handle* satisfies *kind* (cycle-safe; see :func:`satisfies`).

        *kind* maps onto its :class:`~agm.agl.semantics.analyses.DataProperty`
        (:func:`_constraint_kind_to_data_property`); see
        :meth:`_nominal_satisfies_property` for the shared declaration-flag walk.
        """
        result = self._nominal_satisfies_property(
            handle,
            _constraint_kind_to_data_property(kind),
            lambda arg: satisfies(arg, kind, self, bounds),
        )
        # Only registered declarations can acquire later exception descendants.
        if (
            result
            and kind is ConstraintKind.HASHABLE
            and handle not in self._hashable_proofs
            and handle.decl_id in self._defs
            and not contains_type_var(handle)
            and not contains_inference_var(handle)
        ):
            self._hashable_proofs.add(handle)
        return result

    @property
    def hashable_proofs(self) -> frozenset[RecordType | EnumType | ExceptionType]:
        """Concrete nominal Hashable checks already relied on by checked code."""
        return frozenset(self._hashable_proofs)

    def nominal_reaches_non_data(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if a non-data type is reachable from *handle* (cycle-safe).

        The non-data types are ``unit`` and function types. Exactly
        :meth:`nominal_satisfies` for ``Eq``, in open-world mode, negated —
        every type variable *handle* mentions is assumed to satisfy ``Eq``
        since none is in scope here.
        """
        return not self.nominal_satisfies(handle, ConstraintKind.EQ, None)

    def nominal_is_json_convertible(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if *handle* has a JSON representation.

        A record and an exception convert to a JSON object of their fields, an
        enum to its member's tag (with the member's fields, if any has one), so
        the obstacles are a non-data leaf or a non-``Hashable``-keyed ``dict`` somewhere inside
        (:attr:`DataProperty.JSON_CONVERTIBLE`), checked structurally via
        :func:`is_json_convertible` rather than through :func:`satisfies`.
        """
        return self._nominal_satisfies_property(
            handle,
            DataProperty.JSON_CONVERTIBLE,
            lambda arg: is_json_convertible(arg, self),
        )

    def nominal_is_extern_keyable(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if *handle* may cross an extern boundary (cycle-safe).

        Used by :func:`is_extern_keyable` for its nominal case; see
        :attr:`DataProperty.EXTERN_KEYABLE`.
        """
        return self._nominal_satisfies_property(
            handle,
            DataProperty.EXTERN_KEYABLE,
            lambda arg: is_extern_keyable(arg, self),
        )

    def _nominal_satisfies_property(
        self,
        handle: RecordType | EnumType | ExceptionType,
        prop: DataProperty,
        structural: Callable[[Type], bool],
    ) -> bool:
        """Shared declaration-flag + relevant-type-argument walk for one *prop*.

        Declaration-level: *handle*'s declaration must not be flagged
        (:meth:`_declaration_flags`, one fixpoint per *prop*, checked first
        so a flag applying by declaration identity alone — e.g. host-minted
        origin — still applies before any ``TypeDef`` is registered), and,
        when registered, each concrete ``type_args`` entry at a relevant
        parameter position must itself satisfy *prop*, via *structural*; an
        exception carries no ``type_args``, so only its declaration flag
        applies.

        For a policy with ``dict_key_ok`` set (``JSON_CONVERTIBLE``/
        ``EXTERN_KEYABLE``), the argument at a KEY parameter position
        (``DeclarationFlags.key_params`` — e.g. ``Box[K]`` with field
        ``d: dict[K, int]``) must ADDITIONALLY satisfy that policy's own
        ``dict_key_ok`` check directly: a key parameter's own occurrence in
        the declaration's template never flags it by itself (deferred, like
        any bare type variable), so this is the only place that check
        actually happens, at the concrete reference.
        """
        flags = self._declaration_flags(prop)
        if handle.decl_id in flags.flagged:
            return False
        if isinstance(handle, ExceptionType):
            return True
        typedef = self._defs[handle.decl_id]
        relevant = self.relevant_params_by_decl()[handle.decl_id]
        if not all(
            structural(arg)
            for pname, arg in zip(typedef.type_params, handle.type_args)
            if pname in relevant
        ):
            return False
        dict_key_ok = LEAF_POLICIES[prop].dict_key_ok
        if dict_key_ok is None:
            return True
        key_params = flags.key_params[handle.decl_id]
        return all(
            dict_key_ok(arg, self, frozenset())
            for pname, arg in zip(typedef.type_params, handle.type_args)
            if pname in key_params
        )

    def _declaration_flags(self, prop: DataProperty) -> "DeclarationFlags":
        cached = self._declaration_flags_cache.get(prop)
        if cached is None:
            from agm.agl.semantics.analyses import compute_declaration_flags

            cached = compute_declaration_flags(self, LEAF_POLICIES[prop])
            self._declaration_flags_cache[prop] = cached
        return cached

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
        relevant_params = self.relevant_params_by_decl()
        result_id = bfs_first(
            (ref.decl_id for ref in nominal_references_for_schema(t, self._defs, relevant_params)),
            lambda decl_id: caps.successors[decl_id],
            lambda decl_id: decl_id if decl_id in caps.infinite else None,
            key=self._decl_id_sort_key,
        )
        return None if result_id is None else self._defs[result_id]

    def canonical_schema_type(self, t: Type) -> Type:
        """Return *t* with schema-irrelevant nominal type arguments canonicalized.

        Phantom parameters cannot affect a declaration's emitted JSON schema,
        so schema planning must treat instantiations that differ only at those
        positions as the same node. Relevant arguments are canonicalized
        recursively so phantom differences nested inside them are erased too.
        """
        return self._canonical_schema_type(t, self.relevant_params_by_decl())

    def schema_relevant_params(self, decl_id: DeclId) -> frozenset[str]:
        """Return the subset of *decl_id*'s own type parameters that affect its schema.

        A phantom parameter (absent from this set) never reaches a field or
        variant position, directly or through another declaration's own
        relevant parameter, so it cannot affect *decl_id*'s emitted JSON shape;
        an encode-plan template only takes one parameter per relevant name (see
        ``type_schema.build_encode_plan``'s growing-template builder).
        """
        return self.relevant_params_by_decl()[decl_id]

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
            case DictType(key=key, value=value):
                return DictType(
                    self._canonical_schema_type(key, relevant_params),
                    self._canonical_schema_type(value, relevant_params),
                )
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
        typedef = self._defs[t.decl_id]
        relevant = relevant_params[t.decl_id]
        return tuple(
            self._canonical_schema_type(arg, relevant_params) if pname in relevant else UnitType()
            for pname, arg in zip(typedef.type_params, t.type_args)
        )

    def schema_relevant_type_args(self, t: RecordType | EnumType) -> tuple[Type, ...]:
        """Return the canonical type arguments that should appear in schema identity labels."""
        relevant_params = self.relevant_params_by_decl()
        canonical = cast("RecordType | EnumType", self._canonical_schema_type(t, relevant_params))
        typedef = self._defs[t.decl_id]
        relevant = relevant_params[t.decl_id]
        return tuple(
            arg for pname, arg in zip(typedef.type_params, canonical.type_args) if pname in relevant
        )

    def schema_relevant_nominal_references(
        self, t: Type
    ) -> tuple[RecordType | EnumType | ExceptionType, ...]:
        """Return nominal references that can affect *t*'s finite schema."""
        from agm.agl.semantics.analyses import nominal_references_for_schema

        relevant_params = self.relevant_params_by_decl()
        result: list[RecordType | EnumType | ExceptionType] = []
        for ref in nominal_references_for_schema(t, self._defs, relevant_params):
            canonical = self._canonical_schema_type(ref, relevant_params)
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

        Thin formatter over :meth:`json_representation_culprit`: ``None``
        when *t* converts, otherwise its structural culprit rendered into a
        clause for a caller to splice into its own sentence. Shaped like
        :meth:`no_finite_schema_message`, whose culprit search this mirrors.
        """
        culprit = self.json_representation_culprit(t)
        if culprit is None:
            return None
        match culprit:
            case NonDataLeaf(leaf=leaf):
                return f"'{leaf!r}' has no JSON representation"
            case NonDataField(typedef=typedef, field_name=field_name, field_type=field_type):
                return (
                    f"field '{field_name}' of '{qualified_decl_name(typedef)}' has type "
                    f"'{field_type!r}', which has no JSON representation"
                )
            case BadDictKey(key_type=key_type):
                return f"'{key_type!r}' is not Hashable, so it cannot be a dict key"
            case BadDictKeyField(typedef=typedef, field_name=field_name, key_type=key_type):
                return (
                    f"field '{field_name}' of '{qualified_decl_name(typedef)}' has a dict key "
                    f"of type '{key_type!r}', which is not Hashable"
                )
            case BadKeyArgument(
                source=source,
                field_name=field_name,
                target=target,
                param_name=param_name,
                argument=argument,
            ):
                supplier = (
                    f"field '{field_name}' of '{qualified_decl_name(source)}'"
                    if source is not None and field_name is not None
                    else "the type"
                )
                return (
                    f"{supplier} supplies '{argument!r}' for '{qualified_decl_name(target)}'s "
                    f"key parameter '{param_name}', which is not Hashable"
                )
            case FreeTypeVar(name=name):
                return f"type variable '{name}' may stand for a type with no JSON representation"
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def first_non_data_field(self, t: Type) -> NonDataField | None:
        """Return the declaration field that costs *t* its JSON representation.

        Breadth-first from *t*'s own nominal references, so the shallowest
        declaration carrying a non-data field is reported — the most useful
        culprit for a use-site diagnostic — before one reachable only through
        further hops. Follows a declaration's affected field references, an
        exception's ``extends`` base (whose fields are inherited) and its
        affected descendants (a value statically typed as the ancestor may
        hold one at runtime). Never expands an instantiation, so it
        terminates however the declarations recurse. ``None`` when no
        reachable declaration is to blame.
        """

        def own_culprit(typedef: TypeDef) -> NonDataField | None:
            direct = self._own_non_data_field(typedef)
            return None if direct is None else NonDataField(typedef, *direct)

        return self._field_culprit(t, DataProperty.EQ, self.nominal_reaches_non_data, own_culprit)

    def _own_non_data_field(self, typedef: TypeDef) -> tuple[str, Type] | None:
        """Return *typedef*'s first own field whose type structurally reaches non-data."""
        from agm.agl.semantics.analyses import field_templates

        for field_name, template in field_templates(typedef, self._defs):
            if _first_non_data_leaf(template) is not None:
                return field_name, template
        return None

    def _own_dict_key_culprit(self, typedef: TypeDef) -> BadDictKeyField | BadKeyArgument | None:
        """Return *typedef*'s first own field costing it its JSON representation via a bad key.

        Two shapes, tried per field in declaration order: a nominal reference
        in the field supplying a bad argument for ANOTHER declaration's own
        key parameter (:meth:`_key_argument_culprit`) — the more specific,
        pinpointed cause when one applies — or, failing that, the field's own
        type directly holding a non-``Hashable`` dict key
        (:func:`_first_bad_dict_key`), with *typedef*'s own type parameters
        assumed ``Hashable`` (deferred, like any bare type variable, to
        whatever concrete argument a later reference supplies for them) so a
        phantom or deferred parameter is never wrongly named the culprit.
        """
        from agm.agl.semantics.analyses import field_templates, nominal_references

        own_params = frozenset(typedef.type_params)
        for field_name, template in field_templates(typedef, self._defs):
            for ref in nominal_references(template):
                argument_culprit = self._key_argument_culprit(ref, typedef, field_name, own_params)
                if argument_culprit is not None:
                    return argument_culprit
            bad_key = _first_bad_dict_key(template, self, own_params)
            if bad_key is not None:
                return BadDictKeyField(typedef, field_name, bad_key)
        return None

    def _key_argument_culprit(
        self,
        ref: Type,
        source: TypeDef | None,
        field_name: str | None,
        deferred: frozenset[str],
    ) -> BadKeyArgument | None:
        """Return the bad argument *ref* supplies for its own declaration's key parameter, if any.

        *source*/*field_name* name the field of *source* whose type carries
        *ref* (``None``/``None`` when *ref* is itself the type under
        examination, filling its own declaration's key parameter directly) —
        provenance only, no bearing on which arguments are deferred. *ref*'s own
        type parameters are irrelevant here — only *deferred* (normally
        *source*'s own declaration parameters, or, at a top-level search with no
        enclosing field, the examined type's own free type variables) is
        assumed to satisfy the key rule, exactly as :func:`_key_position_ok`
        does for any other key position.
        """
        if not isinstance(ref, (RecordType, EnumType)):
            return None
        target = self._defs[ref.decl_id]
        key_params = self._declaration_flags(DataProperty.JSON_CONVERTIBLE).key_params[ref.decl_id]
        for pname, arg in zip(target.type_params, ref.type_args):
            if pname in key_params and not dict_key_is_hashable(arg, self, deferred):
                return BadKeyArgument(source, field_name, target, pname, arg)
        return None

    def _bad_dict_key_field_culprit(self, t: Type) -> BadDictKeyField | BadKeyArgument | None:
        """Return the declaration field costing *t* its JSON representation via a bad dict key.

        Mirrors :meth:`first_non_data_field`, hunting a bad dict key
        (:meth:`_own_dict_key_culprit`) instead of a non-data leaf.
        """

        def reaches_bad(ref: RecordType | EnumType | ExceptionType) -> bool:
            return not self.nominal_is_json_convertible(ref)

        return self._field_culprit(
            t, DataProperty.JSON_CONVERTIBLE, reaches_bad, self._own_dict_key_culprit
        )

    def json_representation_culprit(self, t: Type) -> JsonRepresentationCulprit | None:
        """Return why *t* has no JSON representation, as a structural culprit.

        ``None`` when *t* converts (:func:`is_json_convertible` is true) or when
        nothing more specific than "this type does not convert" can be said — an
        unresolved inference variable, say, whose real problem is inference
        rather than representation. Otherwise one tagged culprit, tried in this
        order: (1) a non-data leaf reached structurally; (2) a bad argument
        supplied for a dict-key parameter by *t* itself or by any nominal
        handle nested inside it; (3) the declaration field that carries a
        non-data leaf; (4) a non-``Hashable`` dict key reached directly in
        *t*'s own structure; (5) the declaration field that carries a bad
        dict key, or a bad argument it supplies for a nested declaration's key
        parameter; (6) the free type variable that may stand for one. Shaped
        like :meth:`no_finite_schema_message`, whose culprit search this
        mirrors; :meth:`json_representation_obstacle` formats the result.
        """
        from agm.agl.semantics.analyses import nominal_references

        if is_json_convertible(t, self):
            return None
        leaf = _first_non_data_leaf(t)
        if leaf is not None:
            return NonDataLeaf(leaf)
        # A free type variable at a key-argument or direct-key position is
        # deferred here too: its real Hashable-ness depends on whatever concrete
        # type a later reference supplies, so it must fall through to the
        # FreeTypeVar culprit below rather than being wrongly named a bad
        # argument or a bad key outright.
        deferred = free_type_vars(t)
        for ref in nominal_references(t):
            own_argument = self._key_argument_culprit(ref, None, None, deferred)
            if own_argument is not None:
                return own_argument
        field_culprit = self.first_non_data_field(t)
        if field_culprit is not None:
            return field_culprit
        bad_key = _first_bad_dict_key(t, self, deferred)
        if bad_key is not None:
            return BadDictKey(bad_key)
        bad_key_field = self._bad_dict_key_field_culprit(t)
        if bad_key_field is not None:
            return bad_key_field
        type_vars = sorted(deferred)
        if type_vars:
            return FreeTypeVar(type_vars[0])
        return None

    def _field_culprit[R](
        self,
        t: Type,
        prop: DataProperty,
        reaches_bad: Callable[[RecordType | EnumType | ExceptionType], bool],
        own_culprit: Callable[[TypeDef], R],
    ) -> R | None:
        """Breadth-first search from *t*'s own nominal references for the first declaration
        *own_culprit* names.

        Shared by :meth:`first_non_data_field` and :meth:`_bad_dict_key_field_culprit`:
        *prop* selects which declaration-level fixpoint successors follow
        (:meth:`_affected_successors`), *reaches_bad* which references seed
        and expand the search, and *own_culprit* what a visited declaration's
        own fields are checked for (already optional in *R* itself). Breadth-
        first so the shallowest declaration is reported — the most useful
        culprit for a use-site diagnostic — before one reachable only through
        further hops.
        """
        from agm.agl.semantics.analyses import nominal_references

        def culprit(decl_id: DeclId) -> R:
            return own_culprit(self._defs[decl_id])

        def successors(decl_id: DeclId) -> set[DeclId]:
            return self._affected_successors(decl_id, self._defs[decl_id], prop, reaches_bad)

        return bfs_first(
            (ref.decl_id for ref in nominal_references(t) if reaches_bad(ref)),
            successors,
            culprit,
            key=self._decl_id_sort_key,
        )

    def _affected_successors(
        self,
        decl_id: DeclId,
        typedef: TypeDef,
        prop: DataProperty,
        reaches_bad: Callable[[RecordType | EnumType | ExceptionType], bool],
    ) -> set[DeclId]:
        """Return the declarations *typedef* reaches a *prop*-bad declaration through.

        A field reference is a HANDLE, so it is tested with *reaches_bad*,
        which also consults the concrete type arguments — ``Box[agent]`` is
        affected while ``Box`` itself is not. An exception's ``extends`` base
        and its descendants are bare declaration identities carrying no
        arguments, so for them the declaration-level *prop* ``flagged`` set is
        the whole answer. The two idioms below are therefore not
        interchangeable.
        """
        from agm.agl.semantics.analyses import field_templates, nominal_references

        flags = self._declaration_flags(prop).flagged
        result: set[DeclId] = set()
        for _field_name, template in field_templates(typedef, self._defs):
            for ref in nominal_references(template):
                if reaches_bad(ref):
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

    def relevant_params_by_decl(self) -> Mapping[DeclId, frozenset[str]]:
        """Return each declaration's own type parameters that can reach a field.

        Phantom parameters are absent.
        """
        if self._relevant_params is None:
            from agm.agl.semantics.analyses import compute_relevant_params

            self._relevant_params = {
                decl_id: frozenset(params)
                for decl_id, params in compute_relevant_params(
                    self._defs, through_containers=True
                ).items()
            }
        return self._relevant_params

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
            self._forget_methods_of(decl_id)
            self._invalidate_cache_for(decl_id)
        for name_key, decl_id in other._name_index.items():
            self._name_index[name_key] = decl_id
        for enum_id, members in other._inline_members.items():
            self._inline_members.setdefault(enum_id, {}).update(members)
        # Orphan status travels with the declaration: a session seeds a fresh
        # table from its accumulated one on every entry, so a declaration
        # orphaned once must stay orphaned for the rest of the session.
        self._orphaned |= other._orphaned
        self._hashable_proofs.update(other._hashable_proofs)
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
    """Deterministic sort key for registered *decl_id*, resolved to its declaration in *defs*.

    Every SCC/BFS traversal that has to order declaration identities sorts by
    declaration name (:func:`decl_def_sort_key`) rather than by identity, so a
    "first"/"culprit" declaration chosen from a fixpoint never depends on
    declaration numbering.
    """
    return decl_def_sort_key(defs[decl_id])


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


def _first_bad_dict_key(t: Type, table: TypeTable, assume_ok: frozenset[str]) -> Type | None:
    """Return the first non-``Hashable`` dict key type reachable through *t*'s own structure.

    Structural only, like :func:`_first_non_data_leaf`: recurses through
    ``array``/``dict`` but stops at a nominal handle, whose own fields are a
    declaration-level question answered by
    :meth:`TypeTable._bad_dict_key_field_culprit` instead. A dict's own
    (already ``Hashable``) key can never itself embed another dict
    (``Hashable`` excludes ``array``/``dict`` outright), so only the value
    recurses further. *assume_ok* — see :attr:`LeafPolicy.dict_key_ok`;
    passed by a search over one declaration's OWN templates, which assumes
    the declaration's own type parameters satisfy Hashable (deferred to
    whatever concrete argument a later reference supplies for them), so a
    phantom or deferred parameter is never wrongly named the culprit.
    """
    if isinstance(t, DictType):
        if not dict_key_is_hashable(t.key, table, assume_ok):
            return t.key
        return _first_bad_dict_key(t.value, table, assume_ok)
    if isinstance(t, (RecordType, EnumType, ExceptionType)):
        return None
    for child in type_children(t):
        found = _first_bad_dict_key(child, table, assume_ok)
        if found is not None:
            return found
    return None


def satisfies(
    t: Type, kind: ConstraintKind, table: TypeTable, bounds: ConstraintBounds | None
) -> bool:
    """Return ``True`` if a value of type ``t`` satisfies *kind*, given in-scope ``bounds``.

    Structural: every scalar satisfies both kinds; a function or ``unit`` type satisfies
    neither, transitively (a container/record/enum/exception that reaches one
    at any depth is itself disqualified — the nominal case defers to
    :meth:`TypeTable.nominal_satisfies`); ``array`` satisfies only ``Eq``,
    recursing into the element type; ``dict`` satisfies only ``Eq``, and only
    when both its key and value do (never ``Hashable``, regardless of content).

    A bare type variable, the bottom type, and an unresolved inference
    variable — anywhere, including nested inside an array/dict/nominal
    argument — satisfy *kind* when ``bounds`` is ``None`` (open-world mode:
    the bound is deferred to wherever the type is actually instantiated),
    and otherwise only a type variable does, and only when ``bounds`` states
    *kind* for its name (``bounds`` are implication-closed, see
    :func:`~agm.agl.constraints.close_constraints`).
    """
    match t:
        case TypeVarType():
            return bounds is None or kind in bounds.get(t.name, frozenset())
        case BottomType() | InferenceVarType():
            return bounds is None
        case FunctionType() | UnitType():
            return False
        case ArrayType():
            return kind is ConstraintKind.EQ and satisfies(t.elem, kind, table, bounds)
        case DictType():
            return (
                kind is ConstraintKind.EQ
                and satisfies(t.key, kind, table, bounds)
                and satisfies(t.value, kind, table, bounds)
            )
        case RecordType() | EnumType() | ExceptionType():
            return table.nominal_satisfies(t, kind, bounds)
        case TextType() | JsonType() | BoolType() | IntType() | DecimalType():
            return True
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def comparable_types(left: Type, right: Type, table: TypeTable, bounds: ConstraintBounds) -> bool:
    """Return ``True`` if ``left`` and ``right`` may be compared.

    Equality (``=``, ``!=``) and ordering comparisons require both operands to
    have the **same** type after the single ``int → decimal`` widening, or one
    operand's type to widen nominally to the other's (:func:`_nominal_widens`,
    either direction): an enum compares with its members and with a wider
    enum, an exception with its ancestors. Unlike
    :func:`~agm.agl.semantics.types.is_assignable`, ``json`` does **not** absorb
    JSON-shaped scalars here: ``json = json`` is allowed but ``json`` vs any
    non-``json`` type is a static error.

    Thin wrapper over :func:`satisfies` (``Eq``) plus the identity/numeric-pair
    rule. A type variable — top-level or nested — is comparable only when its
    name is bound ``Eq``/``Hashable`` in ``bounds``; the bottom type and an
    inference variable are then never comparable.
    Callers always supply the checker's real bound environment (possibly
    empty), never open-world mode.
    """
    return (
        satisfies(left, ConstraintKind.EQ, table, bounds)
        and satisfies(right, ConstraintKind.EQ, table, bounds)
        and (
            same_comparison_type(left, right)
            or _nominal_widens(table, left, right)
            or _nominal_widens(table, right, left)
        )
    )


def same_comparison_type(left: Type, right: Type) -> bool:
    """Whether ``left`` and ``right`` are one type, or the int/decimal pair."""
    numeric = (IntType, DecimalType)
    return left == right or (isinstance(left, numeric) and isinstance(right, numeric))


# ---------------------------------------------------------------------------
# JSON convertibility and cast classification
# ---------------------------------------------------------------------------


def is_json_convertible(t: Type, table: TypeTable) -> bool:
    """Return ``True`` if ``t`` has a JSON representation.

    The scalars (``text``/``json``/``bool``/``int``/``decimal``) convert
    directly; an ``array`` converts iff its element type does; a ``dict``
    converts iff its key is ``Hashable`` and its value type converts; a
    record or exception
    converts to a JSON object of its fields and an enum
    to its member's tag (with the member's fields, if any has one), so a
    nominal converts iff no non-data type is reachable from its declaration
    (:meth:`TypeTable.nominal_is_json_convertible`). The non-data types —
    ``unit`` and function types — have no representation at all.

    A free type variable is never convertible, its own arm here and, for a
    nominal, in its type arguments: casts are compiled once and type arguments
    are erased, so a ``T`` later instantiated with a function type would
    otherwise reach the conversion at runtime. Note the deliberate asymmetry with the
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
            # A hashable key always has a JSON wire form (text/stringified/
            # entries — chosen at encode/schema time by its DictKeyForm).
            return (
                dict_key_is_hashable(t.key, table)
                and is_json_convertible(t.key, table)
                and is_json_convertible(t.value, table)
            )
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


def is_extern_keyable(t: Type, table: TypeTable) -> bool:
    """Return ``True`` if every ``dict`` in *t* has an extern-keyable key.

    Structural, like :func:`is_json_convertible`, except that a function type
    recurses into its parameter and result types (a companion builds a
    callback's arguments), and a free type variable passes (an extern
    signature's own type parameters are not erased). The only obstacle is a
    ``dict`` whose key is not ``Hashable`` anywhere
    (:class:`~agm.agl.semantics.analyses.DataProperty.EXTERN_KEYABLE`): a
    companion inserts keys into parameter dicts and returns dicts whose keys
    are inserted. A ``dict``'s key is checked directly
    (:func:`dict_key_is_hashable_assuming_type_vars`) and only its value
    recurses; a nominal delegates to the table.
    """
    if isinstance(t, DictType):
        return dict_key_is_hashable_assuming_type_vars(t.key, table) and is_extern_keyable(
            t.value, table
        )
    if isinstance(t, (RecordType, EnumType, ExceptionType)):
        return table.nominal_is_extern_keyable(t)
    return all(is_extern_keyable(child, table) for child in type_children(t))


def is_assignable_in(table: TypeTable, value_type: Type, target_type: Type) -> bool:
    """Return whether a value is assignable to a target in *table*'s nominal context.

    The pure :func:`semantics.types.is_assignable` rules apply unchanged.  In
    addition, a member record is assignable to an enum when it occurs in that
    enum instantiation's declared member set, and an exception is assignable
    to any exception in its base chain. This is deliberately a top-level,
    directed relation: containers remain invariant. An enum is assignable to
    another enum exactly when its constructor set is a subset of the target's.
    """
    return is_assignable(value_type, target_type) or _nominal_widens(table, value_type, target_type)


def _nominal_widens(table: TypeTable, value_type: Type, target_type: Type) -> bool:
    """Return whether *value_type* widens to *target_type* as a member, sub-enum, or subtype."""
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
    # A target is a resolved annotation or ``parse`` argument, never an inference variable.
    target = cast(CheckedType, target)
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

    if isinstance(target, TypeVarType):
        # A cast target that is still a bare type parameter (e.g. `x as T`
        # inside a generic function) names no concrete shape to convert into.
        return CastKind.STATIC_ERROR
    assert_never(target)  # pragma: no cover


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
# prelude types (``ExecResult``, ``Agent``, ``OutputContract``,
# ``OutputContractOption``, ``AgentRequest``, ``SessionTransport``, ``Session``,
# ``SessionStats``, ``SessionError``, ``Sandbox``, ``AgentSandbox``) and the
# generic ``Option`` template.  ``create_seeded_type_table``, the scope resolver's builtin
# constructor-candidate seeding, ``TypeEnvironment`` init seeding, and builtin
# shape validation in the type builder all read these same literals — there
# is exactly one definition of each prelude shape.
# ---------------------------------------------------------------------------


def _standard(fields: tuple[tuple[str, Type], ...]) -> tuple[ParamZone, ...]:
    """Return one ``ParamZone.STANDARD`` per entry of *fields*."""
    return (ParamZone.STANDARD,) * len(fields)


#: A reserved field's host-side default: a scalar, or a nullary enum member by identity.
type ReservedFieldDefault = BoolValue | TextValue | NominalId


def _builtin_enum_defs(
    name: str,
    variants: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...],
    *,
    type_params: tuple[str, ...] = (),
    module_id: ModuleId = RESERVED_ID,
    field_defaults: Mapping[str, ReservedFieldDefault] = MappingProxyType({}),
) -> tuple[TypeDef, tuple[TypeDef, ...]]:
    """Build canonical enum and scoped record-member definitions for the prelude.

    A member field named in *field_defaults* has that declared default; pass
    the same mapping to :func:`_member_field_default_values` for
    :data:`RESERVED_FIELD_DEFAULT_VALUES`.
    """
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
            field_has_default=tuple(field in field_defaults for field, _type in fields),
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


_AGENT_NATIVE_MEMBERS: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...] = (
    ("AgentClaude", (("model", TextType()), ("thinking", TextType()))),
    ("AgentCodex", (("model", TextType()), ("thinking", TextType()))),
    ("AgentPi", (("provider", TextType()), ("model", TextType()), ("thinking", TextType()))),
)
# Every native member field defaults to ``""``; ``AgentCommand``'s command has none.
_AGENT_FIELD_DEFAULTS: Mapping[str, ReservedFieldDefault] = {
    field: TextValue("") for _member, fields in _AGENT_NATIVE_MEMBERS for field, _type in fields
}
_AGENT_DEF, _AGENT_MEMBER_DEFS = _builtin_enum_defs(
    "Agent",
    (("AgentCommand", (("command", TextType()),)), *_AGENT_NATIVE_MEMBERS),
    field_defaults=_AGENT_FIELD_DEFAULTS,
)
_SESSION_TRANSPORT_DEF, _SESSION_TRANSPORT_MEMBER_DEFS = _builtin_enum_defs(
    "SessionTransport", (("Cli", ()), ("Rpc", ()))
)

# ``AgentSandbox``'s ``Disabled``/``Native`` members are inline, fresh
# records scoped under it, like every other builtin enum's members. Its third
# member instead reuses the standalone ``Sandbox`` record's own identity
# (mirroring how ``Optional`` reuses ``Option``'s ``Some``/``None``): the
# value IS a ``Sandbox`` record, so a ``Sandbox`` constructed on its own
# widens into an ``AgentSandbox`` slot with no rewrapping.
_AGENT_SANDBOX_DISABLED_DEF = TypeDef(
    kind="record",
    name="Disabled",
    module_id=RESERVED_ID,
    scope_path=("AgentSandbox",),
    decl_node_id=require_reserved_enum_member_id("AgentSandbox", "Disabled"),
)
_AGENT_SANDBOX_NATIVE_DEF = TypeDef(
    kind="record",
    name="Native",
    module_id=RESERVED_ID,
    scope_path=("AgentSandbox",),
    decl_node_id=require_reserved_enum_member_id("AgentSandbox", "Native"),
)
_AGENT_SANDBOX_MEMBER_DEFS = (_AGENT_SANDBOX_DISABLED_DEF, _AGENT_SANDBOX_NATIVE_DEF)
_AGENT_SANDBOX_DEF = TypeDef(
    kind="enum",
    name="AgentSandbox",
    module_id=RESERVED_ID,
    members=(
        RecordType(
            name="Disabled",
            module_id=RESERVED_ID,
            scope_path=("AgentSandbox",),
            decl_id=require_reserved_enum_member_id("AgentSandbox", "Disabled"),
        ),
        RecordType(
            name="Native",
            module_id=RESERVED_ID,
            scope_path=("AgentSandbox",),
            decl_id=require_reserved_enum_member_id("AgentSandbox", "Native"),
        ),
        RecordType(name="Sandbox", module_id=RESERVED_ID, decl_id=_reserved_id("Sandbox")),
    ),
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
                (
                    "sandbox",
                    EnumType(
                        name="AgentSandbox",
                        module_id=RESERVED_ID,
                        decl_id=_reserved_id("AgentSandbox"),
                    ),
                ),
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
                (
                    "sandbox",
                    EnumType(
                        name="AgentSandbox",
                        module_id=RESERVED_ID,
                        decl_id=_reserved_id("AgentSandbox"),
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
    "Sandbox": TypeDef(
        kind="record",
        name="Sandbox",
        module_id=RESERVED_ID,
        fields=(
            _fields := (
                ("memory", standard_optional_type(TextType())),
                ("swap", standard_optional_type(TextType())),
                ("settings", standard_option_type(TextType())),
                ("patch", BoolType()),
            )
        ),
        field_kinds=_standard(_fields),
        field_has_default=(True, True, True, True),
    ),
    "AgentSandbox": _AGENT_SANDBOX_DEF,
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
        *_AGENT_MEMBER_DEFS,
        *_OUTPUT_CONTRACT_OPTION_MEMBER_DEFS,
        *_SESSION_TRANSPORT_MEMBER_DEFS,
        *_OPTION_MEMBER_DEFS,
        _OPTIONAL_DEFAULT_DEF,
        *_AGENT_SANDBOX_MEMBER_DEFS,
        # ``AgentSandbox``'s ``Sandbox`` member reuses the standalone ``Sandbox``
        # record's own identity (see above), so its member-lookup entry is that
        # same record's own canonical ``TypeDef`` rather than a fresh one.
        BUILTIN_PRELUDE_TYPE_DEFS["Sandbox"],
    )
}

# ---------------------------------------------------------------------------
# Reserved-record field defaults, as host-side constants
#
# An ordinary program's own constructor field default is an ``IrExpr``,
# evaluated once the program is fully linked
# (``IrInterpreter.default_for_field`` against ``NominalDescriptor.field_defaults``
# — see ``runtime.convert.decode_value``'s ``default_resolver``). A host
# engine setting decodes from a CLI flag or config entry *before* any program
# exists, so no evaluator is reachable there; the defaults of ``Sandbox`` and
# of the ``Agent`` members are nevertheless plain constants (``Default``,
# ``None``, ``true``, ``""``). The seeded ``TypeDef`` carries only
# ``field_has_default``; this table holds each default as a constant (a
# nullary enum member by its identity). The lowerer gives the reserved
# descriptors the same constants as IR (``lower.lowerer.reserved_field_defaults``),
# so a program loaded without the standard library constructs these types
# with them too; a standard declaration supersedes the reserved one, so its
# own source defaults win.
# ---------------------------------------------------------------------------


def _member_field_default_values(
    member_defs: tuple[TypeDef, ...], field_defaults: Mapping[str, ReservedFieldDefault]
) -> dict[DeclId, Mapping[int, ReservedFieldDefault]]:
    """Index *field_defaults* by field position for each member declaring one of them."""
    return {
        member.decl_node_id: {
            index: field_defaults[field]
            for index, (field, _type) in enumerate(member.fields)
            if field in field_defaults
        }
        for member in member_defs
        if any(field in field_defaults for field, _type in member.fields)
    }


RESERVED_FIELD_DEFAULT_VALUES: Mapping[DeclId, Mapping[int, ReservedFieldDefault]] = {
    _reserved_id("Sandbox"): {
        0: NominalId(require_reserved_enum_member_id("Optional", "Default")),
        1: NominalId(require_reserved_enum_member_id("Optional", "Default")),
        2: NominalId(require_reserved_enum_member_id("Option", "None")),
        3: BoolValue(True),
    },
    **_member_field_default_values(_AGENT_MEMBER_DEFS, _AGENT_FIELD_DEFAULTS),
}


def reserved_field_default(decl_id: DeclId, field_index: int) -> Value:
    """Return one reserved record's *field_index*'th field's host-side constant default.

    The decode-time default-fill seam for a host engine setting
    (``runtime.engine_config.convert_host_value``'s ``default_resolver``);
    see :data:`RESERVED_FIELD_DEFAULT_VALUES`. *decl_id* names a reserved
    record with defaulted fields (``Sandbox`` or an ``Agent`` member). Keyed
    sparsely by field index, so a field with no default is simply absent
    rather than representable as ``None``.
    """
    match RESERVED_FIELD_DEFAULT_VALUES[decl_id][field_index]:
        case NominalId() as member:
            return RecordValue(nominal=member, fields={})
        case scalar:
            return scalar


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
    "DuplicateKeyError": TypeDef(
        kind="exception",
        name="DuplicateKeyError",
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

    Registers ``BUILTIN_PRELUDE_TYPE_DEFS`` (``ExecResult``,
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
