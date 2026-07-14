"""Shared nominal type-declaration table for AgL.

``RecordType``/``EnumType``/``ExceptionType`` (see ``semantics.types``) are
lightweight handles — ``(module_id, name, type_args)`` for records/enums,
``(module_id, name)`` for exceptions (never generic) — carrying no field/
variant data of their own. This module holds the single source of truth for
their shapes: a table of ``TypeDef`` templates keyed by ``(module_id, name)``,
populated by the type builder as each declaration is resolved.

``TypeDef`` stores field/variant type *templates*: finite ``Type`` trees that
may reference the declaration's own type parameters via ``TypeVarType`` nodes
— the same kind of template already computed for generic types today
(``typecheck.env.GenericTypeDef.template``), just captured under one
representation shared by records, enums, and exceptions.
``TypeTable.record_fields``/``enum_variants`` substitute a handle's
``type_args`` into those templates and memoize the result per handle;
``TypeTable.exception_fields`` has no ``type_args`` to substitute but instead
flattens the ``extends`` base chain into one field mapping.

``comparable_types``/``_has_no_value_equality`` live here rather than in
``semantics.types`` because their record/enum/exception arms consult the
table's declaration-level equality-capability flags instead of walking
embedded fields; ``semantics.types`` cannot import this module without a
circular import. The flags themselves are a fixpoint over the whole table
(``semantics.analyses.compute_equality_capabilities``, cycle-safe by
construction), cached on :class:`TypeTable` and invalidated whenever the
table's declarations change.

:meth:`TypeTable.has_finite_closure`/:meth:`TypeTable.has_finite_schema`
answer a related but distinct whole-type question: not "does this type
support ``=``?" but "is this type's reachable *instantiation closure* finite
(so it has a finite JSON schema)?" — a generic recursive declaration may
reference itself at ever-larger arguments (polymorphic recursion), which
never blocks construction/matching/equality but does mean no finite schema
exists. Backed by ``semantics.analyses.compute_finite_closure``, cached and
invalidated the same way as the equality-capability fixpoint.
:meth:`TypeTable.first_infinite_declaration`/:meth:`TypeTable.no_finite_schema_message`
build on the same query to name the culprit declaration for a use-site
diagnostic (agent output target, cast target, parameter type).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never, cast

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
    InferenceVarType,
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

if TYPE_CHECKING:
    from agm.agl.semantics.analyses import EqualityCapabilities, FiniteClosure

TypeDefKind = Literal["record", "enum", "exception"]

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
    ``base``     — exception metadata: the resolved ``(module_id, name)`` key
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
    """

    kind: TypeDefKind
    name: str
    module_id: ModuleId
    type_params: tuple[str, ...] = ()
    fields: tuple[tuple[str, Type], ...] = ()
    variants: tuple[tuple[str, tuple[tuple[str, Type], ...]], ...] = ()
    abstract: bool = False
    base: tuple[ModuleId, str] | None = None
    field_kinds: tuple[str, ...] = ()

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
        # Exceptions are non-generic, so (unlike record_fields/enum_variants)
        # there is no type_args substitution — the memo is keyed directly by
        # (module_id, name), one entry per exception.
        self._exception_fields_cache: dict[tuple[ModuleId, str], Mapping[str, Type]] = {}
        # Memo for exception_field_kinds — same keying convention as
        # _exception_fields_cache above.
        self._exception_field_kinds_cache: dict[
            tuple[ModuleId, str], tuple[tuple[str, str], ...]
        ] = {}
        # Whole-table equality-capability fixpoint (see
        # :meth:`has_no_value_equality`), computed lazily on first use and
        # invalidated (set back to ``None``) whenever a declaration is added,
        # removed, or overwritten.
        self._equality_caps: EqualityCapabilities | None = None
        # Whole-table finiteness fixpoint (see :meth:`has_finite_schema`),
        # cached and invalidated the same way as ``_equality_caps``.
        self._finite_closure: FiniteClosure | None = None

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
            self._equality_caps = None
            self._finite_closure = None
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
        # Exception field accessors flatten inherited base chains, so changing
        # one exception can invalidate cached descendants as well as the changed
        # key. Clear the exception caches wholesale rather than trying to
        # maintain a reverse-inheritance index.
        self._exception_fields_cache.clear()
        self._exception_field_kinds_cache.clear()
        # The equality-capability and finiteness fixpoints are whole-table
        # (any declaration's flag can in principle depend on any other's), so
        # a single changed key invalidates the whole cached result rather
        # than just this key.
        self._equality_caps = None
        self._finite_closure = None

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

    def exception_fields(self, handle: ExceptionType) -> Mapping[str, Type]:
        """Return *handle*'s fully flattened field types (base chain applied).

        Exceptions are non-generic, so unlike :meth:`record_fields`/
        :meth:`enum_variants` there is no ``type_args`` substitution — the
        result is memoized directly per ``(module_id, name)`` key. Base
        fields come first (the root contributes ``message``/``trace_id``),
        followed by the exception's own fields, matching declaration order.

        Raises ``KeyError`` if no ``TypeDef`` is registered for the handle's
        ``(module_id, name)``. Raises ``AssertionError`` if the registered
        def's ``kind`` is not ``"exception"``, or if the base chain contains
        a cycle — an internal-invariant violation, since the inhabitation
        check (single-module builder post-pass or graph pre-pass) rejects
        ``extends`` cycles as uninhabitable before this can fire in
        production; this guard is for internal robustness, not a user
        diagnostic.
        """
        key = (handle.module_id, handle.name)
        cached = self._exception_fields_cache.get(key)
        if cached is not None:
            return cached
        result = self._flatten_exception_fields(key, _visiting=frozenset())
        self._exception_fields_cache[key] = result
        return result

    def _flatten_exception_fields(
        self, key: tuple[ModuleId, str], *, _visiting: frozenset[tuple[ModuleId, str]]
    ) -> Mapping[str, Type]:
        if key in _visiting:
            raise AssertionError(f"cyclic exception base chain detected at {key!r}")
        typedef = self._require_exception_def(key, caller="exception_fields")
        fields: dict[str, Type] = {}
        if typedef.base is not None:
            fields.update(self._flatten_exception_fields(typedef.base, _visiting=_visiting | {key}))
        fields.update(dict(typedef.fields))
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
        key = (handle.module_id, handle.name)
        cached = self._exception_field_kinds_cache.get(key)
        if cached is not None:
            return cached
        result = self._flatten_exception_field_kinds(key, _visiting=frozenset())
        self._exception_field_kinds_cache[key] = result
        return result

    def _flatten_exception_field_kinds(
        self, key: tuple[ModuleId, str], *, _visiting: frozenset[tuple[ModuleId, str]]
    ) -> tuple[tuple[str, str], ...]:
        if key in _visiting:
            raise AssertionError(f"cyclic exception base chain detected at {key!r}")
        typedef = self._require_exception_def(key, caller="exception_field_kinds")
        inherited: tuple[tuple[str, str], ...] = ()
        if typedef.base is not None:
            inherited = self._flatten_exception_field_kinds(
                typedef.base, _visiting=_visiting | {key}
            )
        own = tuple(
            (fname, kind)
            for (fname, _ftype), kind in zip(typedef.fields, typedef.field_kinds, strict=True)
            if fname != "trace_id"
        )
        return inherited + own

    def exception_def(self, handle: ExceptionType) -> TypeDef:
        """Return the registered ``TypeDef`` for *handle*.

        Used to read exception hierarchy metadata (``abstract``, ``base``)
        that ``ExceptionType`` itself no longer carries. Raises ``KeyError``/
        ``AssertionError`` under the same conditions as
        :meth:`exception_fields`.
        """
        return self._require_exception_def((handle.module_id, handle.name), caller="exception_def")

    def _require_exception_def(self, key: tuple[ModuleId, str], *, caller: str) -> TypeDef:
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
        """Return all registered ``TypeDef``s (used for REPL and graph table sharing)."""
        return tuple(self._defs.values())

    def has_no_value_equality(self, handle: RecordType | EnumType | ExceptionType) -> bool:
        """Return ``True`` if *handle* has no value equality (cycle-safe).

        Declaration-level: *handle*'s declaration is unconditionally
        non-comparable (``EqualityCapabilities.no_equality``), or one of its
        concrete ``type_args`` at an equality-relevant parameter position is
        itself non-comparable — see
        :func:`~agm.agl.semantics.analyses.compute_equality_capabilities` for
        why this reproduces the substitute-then-walk answer without ever
        expanding *handle*'s own fields (so it never re-enters a cycle).
        Exceptions carry no ``type_args``, so only the declaration flag
        applies to them.
        """
        caps = self._equality_capabilities()
        key = (handle.module_id, handle.name)
        if key in caps.no_equality:
            return True
        if isinstance(handle, ExceptionType):
            return False
        typedef = self._defs.get(key)
        if typedef is None:
            return False
        relevant = caps.relevant_params.get(key, frozenset())
        return any(
            _has_no_value_equality(arg, self)
            for pname, arg in zip(typedef.type_params, handle.type_args)
            if pname in relevant
        )

    def _equality_capabilities(self) -> "EqualityCapabilities":
        if self._equality_caps is None:
            from agm.agl.semantics.analyses import compute_equality_capabilities

            self._equality_caps = compute_equality_capabilities(self)
        return self._equality_caps

    def has_finite_closure(self, module_id: ModuleId, name: str) -> bool:
        """Return ``True`` if the declaration ``(module_id, name)`` has a finite closure.

        Declaration-level only (no ``type_args``): see
        :func:`~agm.agl.semantics.analyses.compute_finite_closure` for what
        "finite closure" means and how it is decided. A declaration key that
        is not registered at all defaults to ``True`` (finite), matching the
        defensive default of every other declaration-level query here.
        """
        return (module_id, name) not in self._finite_closure_result().infinite

    def has_finite_schema(self, t: Type) -> bool:
        """Return ``True`` if every declaration reachable from *t* has a finite closure.

        Thin wrapper over :meth:`first_infinite_declaration`: *t* has a
        finite schema iff no infinite declaration is reachable from it.
        """
        return self.first_infinite_declaration(t) is None

    def first_infinite_declaration(self, t: Type) -> tuple[ModuleId, str] | None:
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
        seen: set[tuple[ModuleId, str]] = set()
        queue: deque[tuple[ModuleId, str]] = deque(
            (ref.module_id, ref.name)
            for ref in nominal_references_for_schema(t, self._defs, caps.relevant_params)
        )
        while queue:
            key = queue.popleft()
            if key in seen:
                continue
            seen.add(key)
            if key in caps.infinite:
                return key
            successors: frozenset[tuple[ModuleId, str]] = caps.successors.get(key, frozenset())
            queue.extend(sorted(successors - seen, key=decl_key_sort_key))
        return None

    def canonical_schema_type(self, t: Type) -> Type:
        """Return *t* with schema-irrelevant nominal type arguments canonicalized.

        Phantom parameters cannot affect a declaration's emitted JSON schema,
        so schema planning must treat instantiations that differ only at those
        positions as the same node. Relevant arguments are canonicalized
        recursively so phantom differences nested inside them are erased too.
        """
        return self._canonical_schema_type(t, self._finite_closure_result().relevant_params)

    def _canonical_schema_type(
        self, t: Type, relevant_params: Mapping[tuple[ModuleId, str], frozenset[str]]
    ) -> Type:
        match t:
            case RecordType():
                return RecordType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                )
            case EnumType():
                return EnumType(
                    name=t.name,
                    type_args=self._canonical_schema_args(t, relevant_params),
                    module_id=t.module_id,
                )
            case ListType(elem=elem):
                return ListType(self._canonical_schema_type(elem, relevant_params))
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
        relevant_params: Mapping[tuple[ModuleId, str], frozenset[str]],
    ) -> tuple[Type, ...]:
        key = (t.module_id, t.name)
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
        typedef = self._defs.get((t.module_id, t.name))
        if typedef is None:
            return canonical.type_args
        relevant = caps.relevant_params.get((t.module_id, t.name), frozenset())
        result = [
            arg
            for pname, arg in zip(typedef.type_params, canonical.type_args)
            if pname in relevant
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
        is_own_declaration = isinstance(t, (RecordType, EnumType, ExceptionType)) and (
            t.module_id,
            t.name,
        ) == culprit
        if is_own_declaration:
            return (
                f"type '{t!r}' cannot be used as {use}: its recursive instantiations "
                "never close, so it has no finite JSON schema."
            )
        culprit_module, culprit_name = culprit
        # Qualify the culprit with its module when it is not the entry module
        # (bare names from an imported module can otherwise be ambiguous),
        # matching the ``module.dotted()::name`` convention used by
        # RecordType/EnumType's own ``__repr__``.
        qualified_culprit = (
            culprit_name
            if culprit_module.is_entry
            else f"{culprit_module.dotted()}::{culprit_name}"
        )
        return (
            f"type '{t!r}' cannot be used as {use}: it contains '{qualified_culprit}', whose "
            "recursive instantiations never close, so it has no finite JSON schema."
        )

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
            self._invalidate_cache_for(key)


def decl_key_sort_key(key: tuple[ModuleId, str]) -> tuple[tuple[str, ...], str]:
    """Deterministic sort key for a declaration key (module segments, then name)."""
    return (key[0].segments, key[1])


def _has_no_value_equality(t: Type, table: TypeTable) -> bool:
    """True if ``t`` is, or transitively contains, a type with no value equality.

    Function, agent, and unit values are opaque / identity-only and AgL gives
    them no ``=``/``!=`` operator; ``unit`` has a single value but no equality
    operator.  A list, dict, record, enum, or exception that transitively holds
    such a type is therefore itself not comparable.  ``t`` is always a finite
    tree (list/dict wrapping is structural, not nominal), so recursing through
    ``ListType``/``DictType`` always terminates; a record/enum/exception
    handle instead defers to :meth:`TypeTable.has_no_value_equality`, which
    consults a precomputed declaration-level fixpoint rather than re-walking
    the handle's own fields — the type declarations themselves may be
    recursive, but this function never re-enters them.
    """
    match t:
        case FunctionType() | AgentType() | UnitType():
            return True
        case ListType():
            return _has_no_value_equality(t.elem, table)
        case DictType():
            return _has_no_value_equality(t.value, table)
        case RecordType() | EnumType() | ExceptionType():
            return table.has_no_value_equality(t)
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

_EXCEPTION_ROOT_KEY = (PRELUDE_ID, "Exception")


def _named_only(count: int) -> tuple[str, ...]:
    """Return *count* copies of the ``ParamKind.NAMED_ONLY`` value (one per own field)."""
    return ("named_only",) * count


BUILTIN_EXCEPTION_TYPE_DEFS: Mapping[str, TypeDef] = {
    "Exception": TypeDef(
        kind="exception",
        name="Exception",
        module_id=PRELUDE_ID,
        fields=(("message", TextType()), ("trace_id", TextType())),
        abstract=True,
        field_kinds=_named_only(2),
    ),
    "AgentCallError": TypeDef(
        kind="exception",
        name="AgentCallError",
        module_id=PRELUDE_ID,
        fields=(("agent", TextType()), ("cause", TextType()), ("metadata", JsonType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(3),
    ),
    "AgentParseError": TypeDef(
        kind="exception",
        name="AgentParseError",
        module_id=PRELUDE_ID,
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
        module_id=PRELUDE_ID,
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
        module_id=PRELUDE_ID,
        fields=(("function", TextType()), ("python_type", TextType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "MaxIterationsExceeded": TypeDef(
        kind="exception",
        name="MaxIterationsExceeded",
        module_id=PRELUDE_ID,
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
        module_id=PRELUDE_ID,
        fields=(("scrutinee_type", TextType()), ("scrutinee", JsonType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "IndexError": TypeDef(
        kind="exception",
        name="IndexError",
        module_id=PRELUDE_ID,
        fields=(("index", IntType()), ("length", IntType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "KeyError": TypeDef(
        kind="exception",
        name="KeyError",
        module_id=PRELUDE_ID,
        fields=(("key", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "TypeError": TypeDef(
        kind="exception",
        name="TypeError",
        module_id=PRELUDE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
    "ArithmeticError": TypeDef(
        kind="exception",
        name="ArithmeticError",
        module_id=PRELUDE_ID,
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
        module_id=PRELUDE_ID,
        fields=(("name", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "ImmutableBindingError": TypeDef(
        kind="exception",
        name="ImmutableBindingError",
        module_id=PRELUDE_ID,
        fields=(("name", TextType()), ("operation", TextType())),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(2),
    ),
    "Abort": TypeDef(
        kind="exception",
        name="Abort",
        module_id=PRELUDE_ID,
        base=_EXCEPTION_ROOT_KEY,
    ),
    # AgL: RecursionError raised when the call-depth limit is exceeded.
    "RecursionError": TypeDef(
        kind="exception",
        name="RecursionError",
        module_id=PRELUDE_ID,
        fields=(("limit", IntType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "CastError": TypeDef(
        kind="exception",
        name="CastError",
        module_id=PRELUDE_ID,
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
        module_id=PRELUDE_ID,
        fields=(("raw", TextType()),),
        base=_EXCEPTION_ROOT_KEY,
        field_kinds=_named_only(1),
    ),
    "RangeError": TypeDef(
        kind="exception",
        name="RangeError",
        module_id=PRELUDE_ID,
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
