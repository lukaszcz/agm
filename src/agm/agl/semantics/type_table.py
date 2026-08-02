"""Shared nominal type-declaration table for AgL.

``RecordType``/``EnumType``/``ExceptionType`` (see ``semantics.types``) are
lightweight handles — ``(module_id, scope_path, name, type_args)`` for
records/enums, ``(module_id, scope_path, name)`` for exceptions (never generic)
— carrying no field/variant data of their own. This module holds the single
source of truth for their shapes: a table of ``TypeDef`` templates keyed by
``(module_id, scope_path, name)``,
populated by the type builder as each declaration is resolved.

``TypeDef`` stores field/variant type *templates*: finite ``Type`` trees that
may reference the declaration's own type parameters via ``TypeVarType`` nodes
— the same kind of template already computed for generic types today
(``typecheck.env.GenericTypeDef.template``), just captured under one
representation shared by records, enums, and exceptions.
``TypeTable.record_fields``/``enum_variants`` substitute a handle's
``type_args`` into those templates and memoize the result per handle;
``TypeTable.exception_fields`` has no ``type_args`` to substitute but instead
flattens the ``extends`` base chain into one field mapping. The table also
keeps plain ``MethodDef`` data keyed by nominal owner; exception method lookup
uses the same base-chain flattening and cache discipline as exception fields.

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

:meth:`TypeTable.has_finite_closure`/:meth:`TypeTable.has_finite_schema`
answer a related but distinct whole-type question: not "does this type
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

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, assert_never, cast

from agm.agl.modules.ids import STD_CORE_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.types import (
    AgentType,
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
    TypeVarType,
    UnitType,
    contains_type_var,
    free_type_vars,
    is_assignable,
    is_scalar_json_shaped,
    spells_bare,
    substitute,
    type_children,
)
from agm.util.graph import bfs_first

if TYPE_CHECKING:
    from agm.agl.semantics.analyses import FiniteClosure, NonDataReachability

TypeDefKind = Literal["record", "enum", "exception"]
DeclKey = tuple[ModuleId, tuple[str, ...], str]
NominalOwner = RecordType | EnumType | ExceptionType


@dataclass(frozen=True, slots=True)
class MethodDef:
    """Plain declaration data for one method owned by a nominal type.

    ``module_id``/``scope_path``/``name`` and ``decl_node_id`` identify the
    declared function, not its owner: a root ``Point`` method ``Point::move``
    has declaration scope ``("Point",)`` while its owner key is
    ``(module_id, (), "Point")``.
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


# ``ParamKind.value`` strings (``"positional_only"``/``"standard"``/
# ``"named_only"``) — ``semantics`` may not import ``syntax.nodes`` (see
# ``tests/test_agl_dependencies.py``), so ``TypeDef.field_kinds`` below stores
# the stable string values instead of the ``ParamKind`` enum itself; the
# ``typecheck`` layer (which already imports both) converts back with
# ``ParamKind(value)``.


@dataclass(frozen=True, slots=True)
class TypeDef:
    """One nominal type declaration's parameter list and field/variant templates.

    ``fields``/``variants`` are stored as tuples (not dicts) so ``TypeDef``
    stays hashable and declaration order is explicit; ``TypeTable`` exposes
    mapping-shaped accessors that substitute a handle's ``type_args`` in and
    cache the result.

    ``fields``   — field templates for records (empty for enums); for
                   exceptions, the exception's OWN field templates only —
                   NOT flattened with the base chain (see
                   :meth:`TypeTable.exception_fields`).
    ``variants`` — variant templates for enums: ``(name, fields)`` pairs
                   (empty for records/exceptions).
    ``abstract`` — exception metadata: ``True`` for the hierarchy root
                   (catchable but not constructible); unused for
                   records/enums.
    ``base``     — exception metadata: the resolved ``(module_id, scope_path, name)`` key
                   of the ``extends`` target, or ``None`` for the root;
                   unused for records/enums.
    ``field_kinds`` — exception metadata: the OWN parameter kind (positional-
                   only/standard/named-only, from the declaration's ``@pos``/
                   ``@std``/``@named`` markers) for each entry of ``fields``,
                   in the same order — a field's declared kind is honored the
                   same way a record's is, it is not forced to named-only.
                   Stored as ``ParamKind.value`` strings, not the enum itself
                   (``semantics`` may not import ``syntax.nodes``); see the
                   module-level comment above.  Unused for records/enums,
                   whose constructor kinds live in the separate
                   ``TypeEnvironment`` registry instead. See
                   :meth:`TypeTable.exception_field_kinds`, which flattens
                   this alongside the base chain.
    ``is_builtin`` — ``True`` when this entry came from a source ``builtin``
                   declaration, at whatever path it was written. It is
                   metadata about the declaration, not part of its shape, so
                   it is excluded from equality/hashing (``compare=False``):
                   :meth:`_TypeBuilder._validate_builtin_shape` compares a
                   ``builtin`` declaration's whole ``TypeDef`` against a
                   seeded canonical literal, which never sets this flag.
    """

    kind: TypeDefKind
    name: str
    module_id: ModuleId
    scope_path: tuple[str, ...] = ()
    type_params: tuple[str, ...] = ()
    fields: tuple[tuple[str, Type], ...] = ()
    variants: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...] = ()
    abstract: bool = False
    base: DeclKey | None = None
    field_kinds: tuple[str, ...] = ()
    is_builtin: bool = field(default=False, compare=False)

    def handle(self, type_args: tuple[Type, ...] = ()) -> RecordType | EnumType:
        """Return the ``RecordType``/``EnumType`` handle naming this ``TypeDef``.

        Convenience for call sites that hold a ``TypeDef`` and need the
        corresponding handle (e.g. to register a value, or to pass to
        :meth:`TypeTable.record_fields`/:meth:`TypeTable.enum_variants`).
        *type_args* defaults to ``()`` for non-generic defs.
        """
        if self.kind == "record":
            return RecordType(
                name=self.name,
                type_args=type_args,
                module_id=self.module_id,
                scope_path=self.scope_path,
            )
        if self.kind == "enum":
            return EnumType(
                name=self.name,
                type_args=type_args,
                module_id=self.module_id,
                scope_path=self.scope_path,
            )
        raise ValueError(f"TypeDef.handle() does not support kind {self.kind!r}")


class TypeTable:
    """Mutable registry of ``TypeDef``s keyed by ``(module_id, scope_path, name)``.

    Populated by the type builder as each declaration's body is resolved;
    a single instance is shared
    across a module graph's per-module environments so every module's
    declarations land in the same table.
    """

    def __init__(self) -> None:
        self._defs: dict[DeclKey, TypeDef] = {}
        self._record_fields_cache: dict[DeclKey, dict[RecordType, Mapping[str, Type]]] = {}
        self._enum_variants_cache: dict[
            DeclKey, dict[EnumType, Mapping[str, Mapping[str, Type]]]
        ] = {}
        # Exceptions are non-generic, so (unlike record_fields/enum_variants)
        # there is no type_args substitution — the memo is keyed directly by
        # (module_id, scope_path, name), one entry per exception.
        self._exception_fields_cache: dict[DeclKey, Mapping[str, Type]] = {}
        # Memo for exception_field_kinds — same keying convention as
        # _exception_fields_cache above.
        self._exception_field_kinds_cache: dict[DeclKey, tuple[tuple[str, str], ...]] = {}
        # Methods are independent plain declaration data, keyed by their
        # nominal owner rather than by an import environment. Exception
        # method maps flatten inherited entries and therefore need the same
        # whole-cache invalidation as exception fields.
        self._methods: dict[DeclKey, dict[str, MethodDef]] = {}
        self._exception_methods_cache: dict[DeclKey, Mapping[str, MethodDef]] = {}
        # Whole-table non-data-reachability fixpoint (see
        # :meth:`nominal_reaches_non_data`), computed lazily on first use and
        # invalidated (set back to ``None``) whenever a declaration is added,
        # removed, or overwritten.
        self._non_data_caps: NonDataReachability | None = None
        # Whole-table finiteness fixpoint (see :meth:`has_finite_schema`),
        # cached and invalidated the same way as ``_non_data_caps``.
        self._finite_closure: FiniteClosure | None = None

    def register(self, typedef: TypeDef) -> None:
        """Register *typedef*, idempotent under identical re-registration.

        Registering a *different* definition under an already-registered
        ``(module_id, scope_path, name)`` key is an internal invariant violation — every
        declaration is built exactly once per module, so, when self-validation is
        enabled, this raises ``AssertionError`` rather than a user-facing
        diagnostic. Re-checking the identical declaration again (e.g. the REPL
        re-checking a promoted entry against a fresh environment, or the program
        pre-pass and the per-module check both building the same module) is
        expected and is silently accepted. With self-validation disabled (the
        production path), a re-registration under an existing key is always
        silently accepted, matching declaration reuse rather than re-verifying it.
        """
        if typedef.base is not None and len(typedef.base) == 2:
            typedef = replace(typedef, base=(typedef.base[0], (), typedef.base[1]))
        key = (typedef.module_id, typedef.scope_path, typedef.name)
        existing = self._defs.get(key)
        if existing is None:
            self._defs[key] = typedef
            self._non_data_caps = None
            self._finite_closure = None
            return
        if self_validation_enabled() and existing != typedef:
            raise AssertionError(
                f"conflicting TypeDef registration for {key!r}: "
                f"{existing!r} is already registered, got {typedef!r}"
            )

    def get(
        self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()
    ) -> TypeDef | None:
        """Return the registered ``TypeDef`` for its structured nominal identity."""
        return self._defs.get((module_id, scope_path, name))

    def unregister(self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()) -> None:
        """Remove any registered def for ``(module_id, scope_path, name)``, if present.

        Used when a declaration is about to be redefined (e.g. an incremental
        REPL entry redeclaring an earlier record under the same name with a
        different shape): dropping the stale entry first means the new
        declaration's :meth:`register` call is always a fresh registration,
        never a conflicting one. Also drops any cached substitutions for
        handles under this key, since they were computed from the def being
        removed.
        """
        key = (module_id, scope_path, name)
        self._defs.pop(key, None)
        self._methods.pop(key, None)
        self._invalidate_cache_for(key)

    def _put_method(self, key: DeclKey, method: MethodDef) -> None:
        """Write *method* into *key*'s direct map, invalidating caches on change.

        A repeated registration of the same declaration is a no-op, while a
        redeclaration replaces the previous entry so REPL state cannot retain a
        stale method.
        """
        methods = self._methods.setdefault(key, {})
        if methods.get(method.name) == method:
            return
        methods[method.name] = method
        self._exception_methods_cache.clear()

    def register_method(self, owner: NominalOwner, method: MethodDef) -> None:
        """Register *method* under its nominal *owner*.

        Method declarations are deliberately data-only: scope classifies a
        receiver and typecheck resolves its header before calling this table;
        neither frontend package is imported here.
        """
        self._put_method((owner.module_id, owner.scope_path, owner.name), method)

    def restore_methods_from(
        self,
        other: "TypeTable",
        module_id: ModuleId,
        name: str,
        scope_path: tuple[str, ...] = (),
    ) -> None:
        """Restore one nominal owner's direct method map from *other*."""
        key = (module_id, scope_path, name)
        methods = other._methods.get(key)
        if methods is not None:
            self._methods[key] = dict(methods)
            self._exception_methods_cache.clear()

    def methods_for(self, owner: NominalOwner) -> Mapping[str, MethodDef]:
        """Return methods available on *owner*, including exception bases.

        Record and enum methods are their owner's direct entries. Exception
        methods are base-first flattened mappings, cached per nominal owner;
        the typecheck pass owns the later no-overriding diagnostic, so a direct
        entry wins if invalid data reaches this low-level registry.
        """
        key = (owner.module_id, owner.scope_path, owner.name)
        if not isinstance(owner, ExceptionType):
            return self._methods.get(key, {})
        cached = self._exception_methods_cache.get(key)
        if cached is not None:
            return cached
        result = self._flatten_exception_methods(key)
        self._exception_methods_cache[key] = result
        return result

    def lookup_method(self, owner: NominalOwner, name: str) -> MethodDef | None:
        """Return the available method named *name*, or ``None`` on a miss."""
        return self.methods_for(owner).get(name)

    def declared_methods(self, key: DeclKey) -> Mapping[str, MethodDef]:
        """Return only the methods declared directly on *key*, never inherited ones.

        Declaration-level rules attribute a member to the type that declares it,
        which the inheritance-flattening :meth:`methods_for` cannot answer.
        """
        return self._methods.get(key, {})

    def _exception_chain(self, key: DeclKey, *, caller: str) -> list[tuple[DeclKey, TypeDef]]:
        """Return *key*'s base chain, base first, rejecting a cyclic base link.

        Every flattened exception accessor inherits base-first declaration order
        from this one walk, so ``caller`` only selects the ``KeyError``/
        ``AssertionError`` label of the accessor that asked.
        """
        chain: list[tuple[DeclKey, TypeDef]] = []
        visited: set[DeclKey] = set()
        current: DeclKey | None = key
        while current is not None:
            if current in visited:
                raise AssertionError(f"cyclic exception base chain detected at {current!r}")
            visited.add(current)
            typedef = self._require_exception_def(current, caller=caller)
            chain.append((current, typedef))
            current = typedef.base
        chain.reverse()
        return chain

    def ancestor_defs(self, key: DeclKey) -> tuple[TypeDef, ...]:
        """Return *key*'s exception ancestors, nearest first.

        Empty for a hierarchy root and for any non-exception declaration, so a
        caller checking inherited members needs no base-chain walk of its own.
        """
        typedef = self.get(key[0], key[2], key[1])
        assert typedef is not None, f"no TypeDef registered for {key!r}"
        if typedef.kind != "exception" or typedef.base is None:
            return ()
        chain = self._exception_chain(typedef.base, caller="ancestor_defs")
        return tuple(base_def for _base_key, base_def in reversed(chain))

    def _flatten_exception_methods(self, key: DeclKey) -> Mapping[str, MethodDef]:
        methods: dict[str, MethodDef] = {}
        for chain_key, _typedef in self._exception_chain(key, caller="methods_for"):
            methods.update(self._methods.get(chain_key, {}))
        return methods

    def _invalidate_cache_for(self, key: DeclKey) -> None:
        self._record_fields_cache.pop(key, None)
        self._enum_variants_cache.pop(key, None)
        # Exception field accessors flatten inherited base chains, so changing
        # one exception can invalidate cached descendants as well as the changed
        # key. Clear the exception caches wholesale rather than trying to
        # maintain a reverse-inheritance index.
        self._exception_fields_cache.clear()
        self._exception_field_kinds_cache.clear()
        self._exception_methods_cache.clear()
        # The non-data-reachability and finiteness fixpoints are whole-table
        # (any declaration's flag can in principle depend on any other's), so
        # a single changed key invalidates the whole cached result rather
        # than just this key.
        self._non_data_caps = None
        self._finite_closure = None

    def record_fields(self, handle: RecordType) -> Mapping[str, Type]:
        """Return *handle*'s field types with its ``type_args`` substituted in.

        Memoized per handle: ``RecordType`` equality/hash exclude ``fields``
        (identity is ``(module_id, name, type_args)``), so the same handle
        always maps to the same substituted mapping object. The memo is
        bucketed by ``(module_id, scope_path, name)`` so a single key's invalidation
        (:meth:`unregister`, :meth:`merge_from`) never has to scan entries for
        other keys.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``(module_id, scope_path, name)`` — every valid handle is expected to have one.
        Raises ``AssertionError`` if the registered def's ``kind`` is not
        ``"record"`` — an internal-invariant violation, since a ``RecordType``
        handle only ever names a record declaration.
        """
        key = (handle.module_id, handle.scope_path, handle.name)
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

        Memoized per handle, bucketed by ``(module_id, scope_path, name)`` (see
        :meth:`record_fields`). Raises ``KeyError`` if no ``TypeDef`` is
        registered for the handle's ``(module_id, scope_path, name)``, or
        ``AssertionError`` if the registered def's ``kind`` is not ``"enum"``.
        """
        key = (handle.module_id, handle.scope_path, handle.name)
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

    def exception_fields(self, handle: ExceptionType) -> Mapping[str, Type]:
        """Return *handle*'s fully flattened field types (base chain applied).

        Exceptions are non-generic, so unlike :meth:`record_fields`/
        :meth:`enum_variants` there is no ``type_args`` substitution — the
        result is memoized directly per ``(module_id, scope_path, name)`` key. Base
        fields come first (the root contributes ``message``/``trace_id``),
        followed by the exception's own fields, matching declaration order.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``(module_id, scope_path, name)``. Raises ``AssertionError`` if the registered
        def's ``kind`` is not ``"exception"``, or if the base chain contains
        a cycle — an internal-invariant violation, since the whole-program
        inhabitation pre-pass rejects ``extends`` cycles as uninhabitable
        before this can fire in production; this guard is for internal
        robustness, not a user diagnostic.
        """
        key = (handle.module_id, handle.scope_path, handle.name)
        cached = self._exception_fields_cache.get(key)
        if cached is not None:
            return cached
        result = self._flatten_exception_fields(key)
        self._exception_fields_cache[key] = result
        return result

    def _flatten_exception_fields(self, key: DeclKey) -> Mapping[str, Type]:
        fields: dict[str, Type] = {}
        for _chain_key, typedef in self._exception_chain(key, caller="exception_fields"):
            fields.update(typedef.fields)
        return fields

    def exception_field_kinds(self, handle: ExceptionType) -> tuple[tuple[str, str], ...]:
        """Return *handle*'s fully flattened ``(field_name, ParamKind.value)`` pairs.

        Mirrors :meth:`exception_fields`'s base-chain flattening (base fields
        first, in declaration order, then the exception's own), but carries
        each field's declared parameter kind instead of its type — an
        exception's OWN fields honor their declared ``@pos``/``@std``/
        ``@named`` marker exactly like a record's fields do (see
        ``TypeDef.field_kinds``); only inheritance is exception-specific.
        ``trace_id`` (present only on the hierarchy root) is excluded: it is
        auto-filled at construction time, never supplied by the caller.

        Each kind is a ``ParamKind.value`` string, not the enum itself (see
        the module-level comment on ``TypeDef.field_kinds``); the caller
        (``typecheck.env``) converts back with ``ParamKind(value)``.

        Raises ``KeyError``/``AssertionError`` under the same conditions as
        :meth:`exception_fields`.
        """
        key = (handle.module_id, handle.scope_path, handle.name)
        cached = self._exception_field_kinds_cache.get(key)
        if cached is not None:
            return cached
        result = self._flatten_exception_field_kinds(key)
        self._exception_field_kinds_cache[key] = result
        return result

    def _flatten_exception_field_kinds(self, key: DeclKey) -> tuple[tuple[str, str], ...]:
        return tuple(
            (fname, kind)
            for _chain_key, typedef in self._exception_chain(key, caller="exception_field_kinds")
            for (fname, _ftype), kind in zip(typedef.fields, typedef.field_kinds, strict=True)
            if fname != "trace_id"
        )

    def exception_def(self, handle: ExceptionType) -> TypeDef:
        """Return the registered ``TypeDef`` for *handle*.

        Used to read exception hierarchy metadata (``abstract``, ``base``)
        that ``ExceptionType`` itself no longer carries. Raises ``KeyError``/
        ``AssertionError`` under the same conditions as
        :meth:`exception_fields`.
        """
        return self._require_exception_def(
            (handle.module_id, handle.scope_path, handle.name), caller="exception_def"
        )

    def _require_exception_def(self, key: DeclKey, *, caller: str) -> TypeDef:
        typedef = self._defs.get(key)
        if typedef is None:
            raise KeyError(f"no TypeDef registered for exception {key!r}")
        if typedef.kind != "exception":
            raise AssertionError(
                f"{caller} called for {key!r}, which is registered as kind "
                f"{typedef.kind!r}, not 'exception'"
            )
        return typedef

    def entries(self) -> tuple[TypeDef, ...]:
        """Return all registered ``TypeDef``s (used for REPL and program table sharing)."""
        return tuple(self._defs.values())

    def builtin_declaration(self, name: str) -> TypeDef | None:
        """Return the live registered ``builtin`` declaration named *name*, if any.

        Returns ``None`` for a program that declares no ``builtin`` of that
        name — the caller falls back to the seeded canonical shape, which is
        not itself a declaration.

        ``validate_builtin_declaration_uniqueness`` allows at most one such
        declaration per compile unit, so only REPL accumulation — ``_defs``
        outlives the entry that filled it — can leave several here. The last
        one registered is the one the current source declares; ``_defs``
        never moves an existing key, so a forward scan's last match is it.
        """
        result: TypeDef | None = None
        for typedef in self._defs.values():
            if typedef.is_builtin and typedef.name == name:
                result = typedef
        return result

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
        key = (handle.module_id, handle.scope_path, handle.name)
        if key in caps.reaches_non_data:
            return True
        if isinstance(handle, ExceptionType):
            return False
        typedef = self._defs.get(key)
        if typedef is None:
            return False
        relevant = caps.relevant_params.get(key, frozenset())
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

    def has_finite_closure(
        self, module_id: ModuleId, name: str, scope_path: tuple[str, ...] = ()
    ) -> bool:
        """Return whether the structured declaration key has a finite closure.

        Declaration-level only (no ``type_args``): see
        :func:`~agm.agl.semantics.analyses.compute_finite_closure` for what
        "finite closure" means and how it is decided. A declaration key that
        is not registered at all defaults to ``True`` (finite), matching the
        defensive default of every other declaration-level query here.
        """
        return (module_id, scope_path, name) not in self._finite_closure_result().infinite

    def has_finite_schema(self, t: Type) -> bool:
        """Return ``True`` if every declaration reachable from *t* has a finite closure.

        Thin wrapper over :meth:`first_infinite_declaration`: *t* has a
        finite schema iff no infinite declaration is reachable from it.
        """
        return self.first_infinite_declaration(t) is None

    def first_infinite_declaration(self, t: Type) -> DeclKey | None:
        """Return the first infinite declaration reachable from *t*, or ``None``.

        Walks *t*'s own (finite) type tree for nominal references
        (:func:`~agm.agl.semantics.analyses.nominal_references`), then
        extends to every transitively reachable declaration via the
        (declaration-level, argument-independent) reference graph, breadth-
        first, checking each one's finite-closure flag. Never expands a
        concrete instantiation, so it terminates regardless of how *t*'s
        declarations recurse. Breadth-first (rather than depth-first) and
        ordered deterministically (*t*'s own nominal references in tree
        order, then each further hop sorted by declaration key) so that, when
        *t* itself names an infinite declaration, that declaration — the most
        useful "culprit" for a use-site diagnostic — is reported before any
        declaration reachable only through a nested field.
        """
        from agm.agl.semantics.analyses import nominal_references_for_schema

        caps = self._finite_closure_result()
        return bfs_first(
            (
                (ref.module_id, ref.scope_path, ref.name)
                for ref in nominal_references_for_schema(t, self._defs, caps.relevant_params)
            ),
            lambda key: caps.successors.get(key, frozenset()),
            lambda key: key if key in caps.infinite else None,
            key=decl_key_sort_key,
        )

    def canonical_schema_type(self, t: Type) -> Type:
        """Return *t* with schema-irrelevant nominal type arguments canonicalized.

        Phantom parameters cannot affect a declaration's emitted JSON schema,
        so schema planning must treat instantiations that differ only at those
        positions as the same node. Relevant arguments are canonicalized
        recursively so phantom differences nested inside them are erased too.
        """
        return self._canonical_schema_type(t, self._finite_closure_result().relevant_params)

    def _canonical_schema_type(
        self, t: Type, relevant_params: Mapping[DeclKey, frozenset[str]]
    ) -> Type:
        match t:
            case RecordType():
                return RecordType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                    scope_path=t.scope_path,
                )
            case EnumType():
                return EnumType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                    scope_path=t.scope_path,
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
                | AgentType()
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
        relevant_params: Mapping[DeclKey, frozenset[str]],
    ) -> tuple[Type, ...]:
        key = (t.module_id, t.scope_path, t.name)
        typedef = self._defs.get(key)
        if typedef is None:
            return tuple(self._canonical_schema_type(arg, relevant_params) for arg in t.type_args)
        relevant = relevant_params.get(key, frozenset())
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
        typedef = self._defs.get((t.module_id, t.scope_path, t.name))
        if typedef is None:
            return canonical.type_args
        relevant = caps.relevant_params.get((t.module_id, t.scope_path, t.name), frozenset())
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
            and (
                t.module_id,
                t.scope_path,
                t.name,
            )
            == culprit
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
            key, field_name, field_type = culprit
            return (
                f"field '{field_name}' of '{qualified_decl_name(key)}' has type "
                f"'{field_type!r}', which has no JSON representation"
            )
        type_vars = sorted(free_type_vars(t))
        if type_vars:
            return (
                f"type variable '{type_vars[0]}' may stand for a type with no JSON representation"
            )
        return None

    def first_non_data_field(self, t: Type) -> tuple[DeclKey, str, Type] | None:
        """Return the declaration field that costs *t* its JSON representation.

        The result is ``(declaration key, field name, that field's declared
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

        def culprit(key: DeclKey) -> tuple[DeclKey, str, Type] | None:
            typedef = self._defs.get(key)
            if typedef is None:  # pragma: no cover
                # Unreachable by construction: every key enqueued was first
                # confirmed to reach a non-data type, which a dangling
                # (never-registered) declaration never does. Kept as a
                # defensive guard, matching the dangling-reference handling in
                # the fixpoints themselves, in case that ever stops holding.
                return None
            direct = self._own_non_data_field(typedef)
            return None if direct is None else (key, *direct)

        def successors(key: DeclKey) -> set[DeclKey]:
            typedef = self._defs.get(key)
            return set() if typedef is None else self._affected_successors(key, typedef)

        return bfs_first(
            (
                (ref.module_id, ref.scope_path, ref.name)
                for ref in nominal_references(t)
                if self.nominal_reaches_non_data(ref)
            ),
            successors,
            culprit,
            key=decl_key_sort_key,
        )

    def _own_non_data_field(self, typedef: TypeDef) -> tuple[str, Type] | None:
        """Return *typedef*'s first own field whose type structurally reaches non-data."""
        from agm.agl.semantics.analyses import field_templates

        for field_name, template in field_templates(typedef):
            if _first_non_data_leaf(template) is not None:
                return field_name, template
        return None

    def _affected_successors(self, key: DeclKey, typedef: TypeDef) -> set[DeclKey]:
        """Return the declarations *typedef* reaches non-data through.

        A field reference is a HANDLE, so it is tested with
        :meth:`nominal_reaches_non_data`, which also consults the concrete
        type arguments — ``Box[agent]`` is affected while ``Box`` itself is
        not. An exception's ``extends`` base and its descendants are bare
        declaration keys carrying no arguments, so for them the
        declaration-level ``flags`` set is the whole answer. The two idioms
        below are therefore not interchangeable.
        """
        from agm.agl.semantics.analyses import field_templates, nominal_references

        flags = self._non_data_reachability().reaches_non_data
        result: set[DeclKey] = set()
        for _field_name, template in field_templates(typedef):
            for ref in nominal_references(template):
                if self.nominal_reaches_non_data(ref):
                    result.add((ref.module_id, ref.scope_path, ref.name))
        if typedef.kind == "exception":
            if typedef.base is not None and typedef.base in flags:
                result.add(typedef.base)
            result.update(
                child_key
                for child_key, child in self._defs.items()
                if child.kind == "exception" and child.base == key and child_key in flags
            )
        return result

    def _finite_closure_result(self) -> "FiniteClosure":
        if self._finite_closure is None:
            from agm.agl.semantics.analyses import compute_finite_closure

            self._finite_closure = compute_finite_closure(self)
        return self._finite_closure

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
            self._methods.pop(key, None)
            self._invalidate_cache_for(key)
        for key, methods in other._methods.items():
            for method in methods.values():
                self._put_method(key, method)


def decl_key_sort_key(key: DeclKey) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Deterministic sort key for a declaration key (module, scope path, then name)."""
    return (key[0].segments, key[1], key[2])


def qualified_decl_name(key: DeclKey) -> str:
    """Return *key*'s user-facing name, module-qualified where a reader needs it.

    Bare names from an imported module can otherwise be ambiguous; the
    ``module::name`` convention matches ``RecordType``/``EnumType``'s own
    ``__repr__``, and shares the same bare/qualified decision
    (:func:`~agm.agl.semantics.types.spells_bare`): entry-module declarations
    and the shipped standard library's own built-in names are spelled bare
    because that is how every reader writes them.
    """
    module, scope_path, name = key
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
    if isinstance(t, (UnitType, AgentType, FunctionType)):
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

    The non-data types are function, agent, and ``unit``: function and agent
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
        case FunctionType() | AgentType() | UnitType():
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

    ``AgentType``, ``FunctionType``, and ``UnitType`` operands are
    NON-comparable — using ``=``/``!=``/``<`` on them is a static error. Agents
    have no equality in AgL; function values are opaque.
    This rule is **transitive**: an ``array``, ``dict``, ``record``, ``enum``, or
    ``exception`` that (at any depth) contains a function, agent, or ``unit``
    value likewise has no equality and cannot be compared with ``=``/``!=``.
    ``table`` resolves record/enum field shapes for that transitive walk.
    """
    # Function/agent/unit values — and any container/record/enum that transitively
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
        case ExceptionType():
            return table.nominal_is_json_convertible(t)
        case RecordType() | EnumType():
            return table.nominal_is_json_convertible(t) and not any(
                contains_type_var(arg) for arg in t.type_args
            )
        case (
            UnitType()
            | AgentType()
            | FunctionType()
            | BottomType()
            | TypeVarType()
            | InferenceVarType()
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


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
    if isinstance(source, (UnitType, AgentType, FunctionType)) or isinstance(
        target, (UnitType, AgentType, FunctionType, BottomType)
    ):
        return CastKind.STATIC_ERROR
    # ExceptionType as target is not in the matrix
    if isinstance(target, ExceptionType):
        return CastKind.STATIC_ERROR

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
        # Every data value renders to text. Non-data sources (unit/agent/function)
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
        module_id=STD_CORE_ID,
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
        module_id=STD_CORE_ID,
        variants=(
            ("Abort", ()),
            ("Retry", (("n", IntType()),)),
        ),
    ),
    "OutputContract": TypeDef(
        kind="record",
        name="OutputContract",
        module_id=STD_CORE_ID,
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
        module_id=STD_CORE_ID,
        variants=(
            ("None", ()),
            ("Some", (("value", RecordType(name="OutputContract", module_id=STD_CORE_ID)),)),
        ),
    ),
    "AgentRequest": TypeDef(
        kind="record",
        name="AgentRequest",
        module_id=STD_CORE_ID,
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
# ``Option[text]``/``Option[json]`` prelude constants, so a program loaded
# without the standard library can still resolve ``enum_variants`` on
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

# ---------------------------------------------------------------------------
# Built-in exception shapes — the single source of truth for every entry of
# ``semantics.types.BUILTIN_EXCEPTIONS``.  ``fields`` holds each exception's
# OWN fields only (the root's ``message``/``trace_id`` are NOT repeated on
# every concrete exception — see :meth:`TypeTable.exception_fields`, which
# flattens the ``base`` chain on demand).  ``field_kinds`` is likewise own-
# fields-only; every built-in exception field is NAMED_ONLY (there is no
# ``@pos``/``@std`` source syntax for a Python-literal ``TypeDef``) — see
# :meth:`TypeTable.exception_field_kinds`.
# ---------------------------------------------------------------------------

_EXCEPTION_ROOT_KEY: DeclKey = (STD_CORE_ID, (), "Exception")


def _named_only(count: int) -> tuple[str, ...]:
    """Return *count* copies of the ``ParamKind.NAMED_ONLY`` value (one per own field)."""
    return ("named_only",) * count


BUILTIN_EXCEPTION_TYPE_DEFS: Mapping[str, TypeDef] = {
    "Exception": TypeDef(
        kind="exception",
        name="Exception",
        module_id=STD_CORE_ID,
        fields=(("message", TextType()), ("trace_id", TextType())),
        abstract=True,
        field_kinds=_named_only(2),
    ),
    "AgentCallError": TypeDef(
        kind="exception",
        name="AgentCallError",
        module_id=STD_CORE_ID,
        fields=(("agent", TextType()), ("cause", TextType()), ("metadata", JsonType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(3),
    ),
    "AgentParseError": TypeDef(
        kind="exception",
        name="AgentParseError",
        module_id=STD_CORE_ID,
        fields=(
            ("agent", TextType()),
            ("target_type", TextType()),
            ("expected_schema", JsonType()),
            ("raw", TextType()),
            ("normalized_raw", TextType()),
            ("validation_errors", JsonType()),
            ("attempts", IntType()),
            ("metadata", JsonType()),
        ),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(8),
    ),
    "ExecError": TypeDef(
        kind="exception",
        name="ExecError",
        module_id=STD_CORE_ID,
        fields=(
            ("command", TextType()),
            ("exit_code", IntType()),
            ("stdout", TextType()),
            ("stderr", TextType()),
            ("timed_out", BoolType()),
        ),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(5),
    ),
    # ``python_type`` is the raising Python exception's class name, or empty for
    # a contract violation (no Python exception was involved).
    "ExternError": TypeDef(
        kind="exception",
        name="ExternError",
        module_id=STD_CORE_ID,
        fields=(("function", TextType()), ("python_type", TextType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "MaxIterationsExceeded": TypeDef(
        kind="exception",
        name="MaxIterationsExceeded",
        module_id=STD_CORE_ID,
        fields=(
            ("limit", IntType()),
            ("condition", TextType()),
            ("last_condition_value", BoolType()),
            ("metadata", JsonType()),
        ),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(4),
    ),
    "MatchError": TypeDef(
        kind="exception",
        name="MatchError",
        module_id=STD_CORE_ID,
        fields=(("scrutinee_type", TextType()), ("scrutinee", JsonType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "IndexError": TypeDef(
        kind="exception",
        name="IndexError",
        module_id=STD_CORE_ID,
        fields=(("index", IntType()), ("length", IntType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "KeyError": TypeDef(
        kind="exception",
        name="KeyError",
        module_id=STD_CORE_ID,
        fields=(("key", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "TypeError": TypeDef(
        kind="exception",
        name="TypeError",
        module_id=STD_CORE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
    "ArithmeticError": TypeDef(
        kind="exception",
        name="ArithmeticError",
        module_id=STD_CORE_ID,
        fields=(("operation", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    # Statically prevented by scope/typecheck (assignment to immutable bindings
    # and undeclared names), but still listed as catchable runtime exceptions
    # for any runtime paths that bypass the static passes.
    "UndefinedVariableError": TypeDef(
        kind="exception",
        name="UndefinedVariableError",
        module_id=STD_CORE_ID,
        fields=(("name", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "ImmutableBindingError": TypeDef(
        kind="exception",
        name="ImmutableBindingError",
        module_id=STD_CORE_ID,
        fields=(("name", TextType()), ("operation", TextType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "Abort": TypeDef(
        kind="exception",
        name="Abort",
        module_id=STD_CORE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
    # AgL: RecursionError raised when the call-depth limit is exceeded.
    "RecursionError": TypeDef(
        kind="exception",
        name="RecursionError",
        module_id=STD_CORE_ID,
        fields=(("limit", IntType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "CastError": TypeDef(
        kind="exception",
        name="CastError",
        module_id=STD_CORE_ID,
        fields=(
            ("source_type", TextType()),
            ("target_type", TextType()),
            ("raw", TextType()),
        ),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(3),
    ),
    "JsonParseError": TypeDef(
        kind="exception",
        name="JsonParseError",
        module_id=STD_CORE_ID,
        fields=(("raw", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "RangeError": TypeDef(
        kind="exception",
        name="RangeError",
        module_id=STD_CORE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
    "CyclicValueError": TypeDef(
        kind="exception",
        name="CyclicValueError",
        module_id=STD_CORE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
}


def create_seeded_type_table() -> TypeTable:
    """Return a fresh ``TypeTable`` pre-populated with built-in defs.

    Registers ``BUILTIN_PRELUDE_TYPE_DEFS`` (``ExecResult``, ``ParsePolicy``,
    ``OutputContract``, ``OutputContractOption``, ``AgentRequest``), the
    generic ``OPTION_TYPE_DEF``, and ``BUILTIN_EXCEPTION_TYPE_DEFS`` (every
    entry of ``semantics.types.BUILTIN_EXCEPTIONS``).
    """
    table = TypeTable()
    for typedef in BUILTIN_PRELUDE_TYPE_DEFS.values():
        table.register(typedef)
    table.register(OPTION_TYPE_DEF)
    for typedef in BUILTIN_EXCEPTION_TYPE_DEFS.values():
        table.register(typedef)
    return table
