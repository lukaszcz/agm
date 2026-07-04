"""Shared nominal type-declaration table for AgL.

``RecordType``/``EnumType`` (see ``semantics.types``) still embed their own
field/variant maps directly. This module introduces a second representation
of the same declarations: a table of ``TypeDef`` templates keyed by
``(module_id, name)``, populated by the type builder *alongside* the embedded
representation (dual-write, no behavior change). It is the foundation for
turning nominal types into lightweight handles whose field/variant shapes are
looked up here instead of carried by value.

``TypeDef`` stores field/variant type *templates*: finite ``Type`` trees that
may reference the declaration's own type parameters via ``TypeVarType`` nodes
— the same kind of template already computed for generic types today
(``typecheck.env.GenericTypeDef.template``), just captured under one
representation shared by records, enums, and (eventually) exceptions.
``TypeTable.record_fields``/``enum_variants`` substitute a handle's
``type_args`` into those templates and memoize the result per handle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from agm.agl.modules.ids import STD_CORE_ID, ModuleId
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    EnumType,
    RecordType,
    Type,
    TypeVarType,
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


class TypeTable:
    """Mutable registry of ``TypeDef``s keyed by ``(module_id, name)``.

    Populated by the type builder alongside the embedded
    ``RecordType``/``EnumType`` representation; a single instance is shared
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


def create_seeded_type_table() -> TypeTable:
    """Return a fresh ``TypeTable`` pre-populated with built-in prelude defs.

    Derives ``TypeDef``s for ``BUILTIN_PRELUDE_TYPES`` (``ExecResult``,
    ``ParsePolicy``, ``OutputContract``, ``OutputContractOption``,
    ``AgentRequest``) from the existing embedded constants in
    ``semantics.types``. Built-in *exceptions* are not seeded here —
    ``ExceptionType`` has no ``module_id`` yet.

    Also seeds a generic ``Option`` template under ``STD_CORE_ID`` (type
    parameter ``T``, variants ``None``/``Some(value: T)``), matching the
    shape of the concrete ``Option[text]``/``Option[json]`` prelude
    constants, so single-module runs without the stdlib module graph can
    still resolve ``enum_variants`` on ``Option`` handles.
    """
    table = TypeTable()
    for name, typ in BUILTIN_PRELUDE_TYPES.items():
        if isinstance(typ, RecordType):
            table.register(
                TypeDef(
                    kind="record",
                    name=name,
                    module_id=typ.module_id,
                    fields=tuple(typ.fields.items()),
                )
            )
            continue
        enum_typ = cast(EnumType, typ)
        table.register(
            TypeDef(
                kind="enum",
                name=name,
                module_id=enum_typ.module_id,
                variants=tuple(
                    (vname, tuple(vfields.items())) for vname, vfields in enum_typ.variants.items()
                ),
            )
        )
    table.register(
        TypeDef(
            kind="enum",
            name="Option",
            module_id=STD_CORE_ID,
            type_params=("T",),
            variants=(
                ("None", ()),
                ("Some", (("value", TypeVarType("T")),)),
            ),
        )
    )
    return table
