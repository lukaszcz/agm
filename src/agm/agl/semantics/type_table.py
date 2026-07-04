"""Shared nominal type-declaration table for AgL.

``RecordType``/``EnumType`` (see ``semantics.types``) are lightweight
handles — ``(module_id, name, type_args)`` — carrying no field/variant data
of their own. This module holds the single source of truth for their
shapes: a table of ``TypeDef`` templates keyed by ``(module_id, name)``,
populated by the type builder as each declaration is resolved.

``TypeDef`` stores field/variant type *templates*: finite ``Type`` trees that
may reference the declaration's own type parameters via ``TypeVarType`` nodes
— the same kind of template already computed for generic types today
(``typecheck.env.GenericTypeDef.template``), just captured under one
representation shared by records, enums, and (eventually) exceptions.
``TypeTable.record_fields``/``enum_variants`` substitute a handle's
``type_args`` into those templates and memoize the result per handle.

``comparable_types``/``_has_no_value_equality`` live here rather than in
``semantics.types`` because their record/enum arms recurse through the table
instead of through embedded fields; ``semantics.types`` cannot import this
module without a circular import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, assert_never

from agm.agl.modules.ids import PRELUDE_ID, STD_CORE_ID, ModuleId
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
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
    substitute,
)

TypeDefKind = Literal["record", "enum", "exception"]


@dataclass(frozen=True, slots=True)
class TypeDef:
    """One nominal type declaration's parameter list and field/variant templates.

    ``fields``/``variants`` are stored as tuples (not dicts) so ``TypeDef``
    stays hashable and declaration order is explicit; ``TypeTable`` exposes
    mapping-shaped accessors that substitute a handle's ``type_args`` in and
    cache the result.

    ``fields``   — field templates for records/exceptions (empty for enums).
    ``variants`` — variant templates for enums: ``(name, fields)`` pairs
                   (empty for records/exceptions).
    ``abstract`` / ``base`` — exception metadata, unused until exceptions are
                   registered here.
    """

    kind: TypeDefKind
    name: str
    module_id: ModuleId
    type_params: tuple[str, ...] = ()
    fields: tuple[tuple[str, Type], ...] = ()
    variants: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...] = ()
    abstract: bool = False
    base: str | None = None

    def handle(self, type_args: tuple[Type, ...] = ()) -> RecordType | EnumType:
        """Return the ``RecordType``/``EnumType`` handle naming this ``TypeDef``.

        Convenience for call sites that hold a ``TypeDef`` and need the
        corresponding handle (e.g. to register a value, or to pass to
        :meth:`TypeTable.record_fields`/:meth:`TypeTable.enum_variants`).
        *type_args* defaults to ``()`` for non-generic defs.
        """
        if self.kind == "record":
            return RecordType(name=self.name, type_args=type_args, module_id=self.module_id)
        if self.kind == "enum":
            return EnumType(name=self.name, type_args=type_args, module_id=self.module_id)
        raise ValueError(f"TypeDef.handle() does not support kind {self.kind!r}")


class TypeTable:
    """Mutable registry of ``TypeDef``s keyed by ``(module_id, name)``.

    Populated by the type builder as each declaration's body is resolved;
    a single instance is shared
    across a module graph's per-module environments so every module's
    declarations land in the same table.
    """

    def __init__(self) -> None:
        self._defs: dict[tuple[ModuleId, str], TypeDef] = {}
        self._record_fields_cache: dict[
            tuple[ModuleId, str], dict[RecordType, Mapping[str, Type]]
        ] = {}
        self._enum_variants_cache: dict[
            tuple[ModuleId, str], dict[EnumType, Mapping[str, Mapping[str, Type]]]
        ] = {}

    def register(self, typedef: TypeDef) -> None:
        """Register *typedef*, idempotent under identical re-registration.

        Registering a *different* definition under an already-registered
        ``(module_id, name)`` key is an internal invariant violation — every
        declaration is built exactly once per module, so this raises
        ``AssertionError`` rather than a user-facing diagnostic. Re-checking
        the identical declaration again (e.g. the REPL re-checking a promoted
        entry against a fresh environment, or the graph pre-pass and the
        per-module check both building the same module) is expected and is
        silently accepted.
        """
        key = (typedef.module_id, typedef.name)
        existing = self._defs.get(key)
        if existing is None:
            self._defs[key] = typedef
            return
        if existing != typedef:
            raise AssertionError(
                f"conflicting TypeDef registration for {key!r}: "
                f"{existing!r} is already registered, got {typedef!r}"
            )

    def get(self, module_id: ModuleId, name: str) -> TypeDef | None:
        """Return the registered ``TypeDef`` for ``(module_id, name)``, or ``None``."""
        return self._defs.get((module_id, name))

    def unregister(self, module_id: ModuleId, name: str) -> None:
        """Remove any registered def for ``(module_id, name)``, if present.

        Used when a declaration is about to be redefined (e.g. an incremental
        REPL entry redeclaring an earlier record under the same name with a
        different shape): dropping the stale entry first means the new
        declaration's :meth:`register` call is always a fresh registration,
        never a conflicting one. Also drops any cached substitutions for
        handles under this key, since they were computed from the def being
        removed.
        """
        key = (module_id, name)
        self._defs.pop(key, None)
        self._invalidate_cache_for(key)

    def _invalidate_cache_for(self, key: tuple[ModuleId, str]) -> None:
        self._record_fields_cache.pop(key, None)
        self._enum_variants_cache.pop(key, None)

    def record_fields(self, handle: RecordType) -> Mapping[str, Type]:
        """Return *handle*'s field types with its ``type_args`` substituted in.

        Memoized per handle: ``RecordType`` equality/hash exclude ``fields``
        (identity is ``(module_id, name, type_args)``), so the same handle
        always maps to the same substituted mapping object. The memo is
        bucketed by ``(module_id, name)`` so a single key's invalidation
        (:meth:`unregister`, :meth:`merge_from`) never has to scan entries for
        other keys.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``(module_id, name)`` — every valid handle is expected to have one.
        Raises ``AssertionError`` if the registered def's ``kind`` is not
        ``"record"`` — an internal-invariant violation, since a ``RecordType``
        handle only ever names a record declaration.
        """
        key = (handle.module_id, handle.name)
        bucket = self._record_fields_cache.get(key)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._defs.get(key)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for record {key!r}")
        if typedef.kind != "record":
            raise AssertionError(
                f"record_fields called for {key!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'record'"
            )
        subst = dict(zip(typedef.type_params, handle.type_args))
        result: Mapping[str, Type] = {
            fname: substitute(ftype, subst) for fname, ftype in typedef.fields
        }
        self._record_fields_cache.setdefault(key, {})[handle] = result
        return result

    def enum_variants(self, handle: EnumType) -> Mapping[str, Mapping[str, Type]]:
        """Return *handle*'s variant field types with its ``type_args`` substituted in.

        Memoized per handle, bucketed by ``(module_id, name)`` (see
        :meth:`record_fields`). Raises ``KeyError`` if no ``TypeDef`` is
        registered for the handle's ``(module_id, name)``, or
        ``AssertionError`` if the registered def's ``kind`` is not ``"enum"``.
        """
        key = (handle.module_id, handle.name)
        bucket = self._enum_variants_cache.get(key)
        if bucket is not None:
            cached = bucket.get(handle)
            if cached is not None:
                return cached
        typedef = self._defs.get(key)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for enum {key!r}")
        if typedef.kind != "enum":
            raise AssertionError(
                f"enum_variants called for {key!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'enum'"
            )
        subst = dict(zip(typedef.type_params, handle.type_args))
        result: Mapping[str, Mapping[str, Type]] = {
            vname: {fname: substitute(ftype, subst) for fname, ftype in vfields}
            for vname, vfields in typedef.variants
        }
        self._enum_variants_cache.setdefault(key, {})[handle] = result
        return result

    def entries(self) -> tuple[TypeDef, ...]:
        """Return all registered ``TypeDef``s (used for REPL and graph table sharing)."""
        return tuple(self._defs.values())

    def merge_from(self, other: "TypeTable") -> None:
        """Copy every entry from *other* into this table.

        Used to carry accumulated declarations across REPL entries (and to
        seed a fresh per-entry environment from the session's persisted
        state). *other* is treated as authoritative: an entry already present
        under the same key is overwritten, mirroring the last-write-wins
        semantics already used to seed the embedded type dict (``_types``).
        A name redeclared with a different shape in the environment being
        seeded is always subsequently rebuilt by the type builder's
        unregister-then-rebuild dance, so a transient overwrite here is never
        left stale in a way that affects final behavior.

        Skips the write (and the resulting cache invalidation) entirely when
        the incoming def is identical to the one already registered under
        that key, since no cached substitution can be stale in that case.
        """
        for key, typedef in other._defs.items():
            if self._defs.get(key) == typedef:
                continue
            self._defs[key] = typedef
            self._invalidate_cache_for(key)


def _has_no_value_equality(t: Type, table: TypeTable) -> bool:
    """True if ``t`` is, or transitively contains, a type with no value equality.

    Function, agent, and unit values are opaque / identity-only and AgL gives
    them no ``=``/``!=`` operator; ``unit`` has a single value but no equality
    operator.  A list, dict, record, enum, or exception that transitively holds
    such a type is therefore itself not comparable.  Record and enum handles are
    walked through *table* (``record_fields``/``enum_variants``); exceptions
    still carry their fields embedded, so that arm walks them directly.
    Recursive types are rejected, so this recursion terminates — the walk
    relies on the declaration graph being acyclic.
    """
    match t:
        case FunctionType() | AgentType() | UnitType():
            return True
        case ListType():
            return _has_no_value_equality(t.elem, table)
        case DictType():
            return _has_no_value_equality(t.value, table)
        case RecordType():
            return any(
                _has_no_value_equality(ft, table) for ft in table.record_fields(t).values()
            )
        case EnumType():
            return any(
                _has_no_value_equality(ft, table)
                for variant in table.enum_variants(t).values()
                for ft in variant.values()
            )
        case ExceptionType():
            return any(_has_no_value_equality(ft, table) for ft in t.fields.values())
        case (TextType() | JsonType() | BoolType() | IntType() | DecimalType()
              | BottomType() | TypeVarType()):
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

    ``AgentType``, ``FunctionType``, and ``UnitType`` operands are
    NON-comparable — using ``=``/``!=``/``<`` on them is a static error. Agents
    have no equality in AgL; function values are opaque.
    This rule is **transitive**: a ``list``, ``dict``, ``record``, ``enum``, or
    ``exception`` that (at any depth) contains a function, agent, or ``unit``
    value likewise has no equality and cannot be compared with ``=``/``!=``.
    ``table`` resolves record/enum field shapes for that transitive walk.
    """
    # Function/agent/unit values — and any container/record/enum that transitively
    # holds one — have no value equality.
    if _has_no_value_equality(left, table) or _has_no_value_equality(right, table):
        return False
    # Bare type variables and the bottom type are never comparable here (the
    # checker additionally rejects bare type variables at the comparison site).
    if isinstance(left, (BottomType, TypeVarType)) or isinstance(right, (BottomType, TypeVarType)):
        return False
    if left == right:
        return True
    # The only cross-type comparison is numeric int↔decimal (either direction).
    numeric = (IntType, DecimalType)
    return isinstance(left, numeric) and isinstance(right, numeric)


# ---------------------------------------------------------------------------
# Prelude type shapes — the single source of truth for built-in nominal types
#
# These ``TypeDef`` literals are the canonical shapes for AgL's built-in
# prelude types (``ExecResult``, ``ParsePolicy``, ``OutputContract``,
# ``OutputContractOption``, ``AgentRequest``) and the generic ``Option``
# template.  ``create_seeded_type_table``, the scope resolver's builtin
# constructor-candidate seeding, ``TypeEnvironment`` init seeding, and builtin
# shape validation in the type builder all read these same literals — there
# is exactly one definition of each prelude shape.
# ---------------------------------------------------------------------------

BUILTIN_PRELUDE_TYPE_DEFS: Mapping[str, TypeDef] = {
    "ExecResult": TypeDef(
        kind="record",
        name="ExecResult",
        module_id=PRELUDE_ID,
        fields=(
            ("stdout", TextType()),
            ("exit_code", IntType()),
            ("stderr", TextType()),
            ("timed_out", BoolType()),
        ),
    ),
    "ParsePolicy": TypeDef(
        kind="enum",
        name="ParsePolicy",
        module_id=PRELUDE_ID,
        variants=(
            ("Abort", ()),
            ("Retry", (("n", IntType()),)),
        ),
    ),
    "OutputContract": TypeDef(
        kind="record",
        name="OutputContract",
        module_id=PRELUDE_ID,
        fields=(
            ("target_type", TextType()),
            ("codec_name", TextType()),
            ("strict_json", JsonType()),
            ("format_instructions", TextType()),
            ("json_schema", JsonType()),
            ("structured_exec", BoolType()),
        ),
    ),
    "OutputContractOption": TypeDef(
        kind="enum",
        name="OutputContractOption",
        module_id=PRELUDE_ID,
        variants=(
            ("None", ()),
            ("Some", (("value", RecordType(name="OutputContract", module_id=PRELUDE_ID)),)),
        ),
    ),
    "AgentRequest": TypeDef(
        kind="record",
        name="AgentRequest",
        module_id=PRELUDE_ID,
        fields=(
            ("agent", TextType()),
            ("prompt", TextType()),
            (
                "target_type",
                EnumType(name="Option", type_args=(TextType(),), module_id=STD_CORE_ID),
            ),
            (
                "format_instructions",
                EnumType(name="Option", type_args=(TextType(),), module_id=STD_CORE_ID),
            ),
            (
                "json_schema",
                EnumType(name="Option", type_args=(JsonType(),), module_id=STD_CORE_ID),
            ),
            ("attempt", IntType()),
            (
                "previous_error",
                EnumType(name="Option", type_args=(TextType(),), module_id=STD_CORE_ID),
            ),
            ("metadata", JsonType()),
        ),
    ),
}

# Generic ``Option`` template under ``STD_CORE_ID`` (type parameter ``T``,
# variants ``None``/``Some(value: T)``), matching the shape of the concrete
# ``Option[text]``/``Option[json]`` prelude constants, so single-module runs
# without the stdlib module graph can still resolve ``enum_variants`` on
# ``Option`` handles.
OPTION_TYPE_DEF = TypeDef(
    kind="enum",
    name="Option",
    module_id=STD_CORE_ID,
    type_params=("T",),
    variants=(
        ("None", ()),
        ("Some", (("value", TypeVarType("T")),)),
    ),
)


def create_seeded_type_table() -> TypeTable:
    """Return a fresh ``TypeTable`` pre-populated with built-in prelude defs.

    Registers ``BUILTIN_PRELUDE_TYPE_DEFS`` (``ExecResult``, ``ParsePolicy``,
    ``OutputContract``, ``OutputContractOption``, ``AgentRequest``) and the
    generic ``OPTION_TYPE_DEF``.  Built-in *exceptions* are not seeded here —
    ``ExceptionType`` has no ``module_id`` yet.
    """
    table = TypeTable()
    for typedef in BUILTIN_PRELUDE_TYPE_DEFS.values():
        table.register(typedef)
    table.register(OPTION_TYPE_DEF)
    return table
