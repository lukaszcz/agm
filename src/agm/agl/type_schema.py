"""Compile-time JSON Schema, decode-schema, and encode-plan derivation.

:func:`derive_schema` produces a JSON Schema ``dict[str, object]`` from a
semantic :class:`~agm.agl.semantics.types.Type`. Every entry point in this
module takes an explicit :class:`~agm.agl.semantics.type_table.TypeTable` and
resolves record fields and enum member sets through it
(``table.record_fields``/``table.enum_members``) rather than through the
``RecordType``/``EnumType`` handle's own embedded maps — the handle carries
only its declaration identity.  The derived schema is used:

1. Embedded in ``OutputContract.format_instructions`` (pretty-printed) so the
   agent receives the precise shape, and as ``OutputContract.json_schema`` so
   API-backed agents can request native structured output.
2. For schema validation via the ``jsonschema`` library inside
   :class:`~agm.agl.runtime.codec.JsonCodec`.

:func:`build_decode_schema` compiles a ``Type`` into a typeless
:class:`~agm.agl.ir.contracts.DecodePlan` (a
:class:`~agm.agl.ir.contracts.DecodeSchema` root plus its ``$defs`` table)
used by the IR evaluator to reconstruct typed ``Value`` objects from
validated JSON without holding checker ``Type`` references.
:func:`derive_schema_and_decode` derives both from one shared recursion plan
for call sites that need both back-to-back.

Derivation rules:
- ``text``    → ``{"type": "string"}``
- ``int``     → ``{"type": "integer"}``
- ``decimal`` → ``{"type": "number"}``
- ``bool``    → ``{"type": "boolean"}``
- ``json``    → ``{}``  (permissive — accepts any JSON value)
- ``array[T]`` → ``{"type": "array", "items": <schema for T>}``
- ``dict[text, V]`` → ``{"type": "object", "additionalProperties": <schema for V>}``
- ``record``  → object schema with ``additionalProperties: false``, ``required``,
                and per-field ``properties``.
- ``enum``    → ``{"oneOf": [...]}`` — one variant schema per variant, each an
                object with a ``"$case"`` const property and any payload fields.

Recursive types: ``derive_schema`` and ``build_decode_schema`` both expand the
concrete *instantiation graph* reachable from *typ* (nodes are concrete
``RecordType``/``EnumType`` handles, edges are the nominal handles occurring
in a node's own substituted fields/variants, memoized on handle equality) and
find its strongly-connected components — computed ONCE per call as a shared
``_SchemaPlan`` (see ``_plan_schema``; ``derive_schema_and_decode`` computes it
only once even when both derivations are needed). An instantiation is
*recursive for this root* iff it sits in a non-trivial component or has a
self-loop; every such instantiation gets one entry under a top-level
``"$defs"`` object (JSON Schema) / ``DecodePlan.defs`` table (decode schema),
keyed identically in both by a sanitized, collision-free name derived from its
display form, and every occurrence of it — including the root itself, if
recursive — is emitted as ``{"$ref": "#/$defs/<key>"}`` / ``RefDecode(key)``
instead of inlined. Non-recursive types have no reachable recursive
instantiation, so no ``"$defs"``/``defs`` entry is added and the output is
unchanged from a purely-inlining derivation. Guarded by
``type_table.has_finite_schema``: a type whose instantiation closure is
infinite (growing polymorphic recursion) has no finite schema to derive at
all; callers are expected to reject such types before reaching this module
(see the use-site checks in ``typecheck/builtins.py``/``typecheck/checker.py``),
so reaching the guard here is an internal-invariant violation, not a normal
user-facing error path.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import assert_never

from agm.agl.ir.contracts import (
    ArrayDecode,
    ArrayEncode,
    DecodePlan,
    DecodeSchema,
    DictDecode,
    DictEncode,
    EncodeDefinition,
    EncodePlan,
    EncodeSchema,
    EnumDecode,
    EnumEncode,
    ExceptionEncode,
    ParamDecoder,
    RecordDecode,
    RecordEncode,
    RefDecode,
    RefEncode,
    ScalarDecode,
    ScalarEncode,
    ScalarKind,
    TypeParameterEncode,
    VariantDecode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    ArrayType,
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
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.util.graph import sccs

# A concrete nominal instantiation — a graph node in the instantiation graph
# below. Record/enum equality includes type_args; exceptions are non-generic.
Instantiation = RecordType | EnumType | ExceptionType


def derive_schema(typ: Type, type_table: TypeTable) -> dict[str, object]:
    """Derive a JSON Schema from a semantic AgL *typ*.

    The returned dictionary is a valid JSON Schema object.  ``Decimal`` and
    ``int`` values round-trip correctly through JSON Schema validation (both
    are acceptable for ``"type": "number"``; ``"type": "integer"`` accepts
    only whole numbers).  *type_table* resolves record/enum field and variant
    shapes for a ``RecordType``/``EnumType`` *typ* (or one nested inside it).

    Recursive instantiations reachable from *typ* (see the module docstring)
    are emitted once under a top-level ``"$defs"`` object and referenced via
    ``{"$ref": "#/$defs/<key>"}`` everywhere they occur, including *typ*
    itself; a non-recursive *typ* gets no ``"$defs"`` key at all, so its
    output is identical to a plain inlining derivation.

    :raises TypeError: if *typ* is an ``ExceptionType`` (exceptions are not
        wire-serialised and have no JSON Schema), or if *typ* has no finite
        JSON schema at all (callers are expected to reject such types before
        calling this function — see ``TypeTable.has_finite_schema``).
    """
    if isinstance(typ, ExceptionType):
        raise TypeError(
            f"ExceptionType {typ.name!r} has no JSON Schema; exceptions are not "
            "wire-serialised by the JSON codec."
        )
    _require_finite_schema(typ, type_table, "derive a JSON Schema")
    plan = _plan_schema(typ, type_table)
    return _emit_schema_with_plan(typ, type_table, plan)


def _require_finite_schema(typ: Type, type_table: TypeTable, action: str) -> None:
    """Raise ``TypeError`` if *typ*'s reachable instantiation closure is infinite.

    Shared guard for :func:`derive_schema`, :func:`build_decode_schema`, and
    :func:`derive_schema_and_decode`: a type whose recursive instantiations
    never close has no finite schema/decode walk to derive at all. Callers
    are expected to reject such types at the use site (JSON-decoded agent
    output target, fallible cast target, parameter type, extern signature — see
    ``typecheck/checker.py`` and ``typecheck/builtins.py``), so reaching this
    guard is an internal-invariant violation, not a normal user-facing error path.
    """
    if not type_table.has_finite_schema(typ):
        raise TypeError(
            f"cannot {action} for {typ!r}: its recursive instantiations "
            "never close, so it has no finite schema. Callers must reject such types "
            "at the use site (see TypeTable.has_finite_schema) before calling "
            "derive_schema/build_decode_schema."
        )


def _emit_schema_with_plan(
    typ: Type, type_table: TypeTable, plan: "_SchemaPlan"
) -> dict[str, object]:
    """Emit *typ*'s JSON Schema (with ``$defs`` if *plan* has any) from an already-built plan."""
    schema = _emit(typ, type_table, plan)
    if plan.keys:
        schema = dict(schema)
        schema["$defs"] = {
            plan.keys[handle]: _emit_body(handle, type_table, plan) for handle in plan.order
        }
    return schema


def derive_schema_and_decode(
    typ: Type, type_table: TypeTable
) -> tuple[dict[str, object], DecodePlan]:
    """Derive both the JSON Schema and the decode plan for *typ* from ONE shared recursion plan.

    Equivalent to calling :func:`derive_schema` and :func:`build_decode_schema`
    separately — same results — but computes the instantiation-graph plan
    (:func:`_plan_schema`) only once. Use this at call sites that need both
    derivations back-to-back (the lowerer's ask/exec contract building,
    :func:`build_param_decoder`, ``JsonCodec.make_contract``) rather than
    calling the two public functions in sequence.
    """
    if isinstance(typ, ExceptionType):
        raise TypeError(
            f"ExceptionType {typ.name!r} has no JSON Schema; exceptions are not "
            "wire-serialised by the JSON codec."
        )
    _require_finite_schema(typ, type_table, "derive a JSON Schema/decode plan")
    plan = _plan_schema(typ, type_table)
    return _emit_schema_with_plan(typ, type_table, plan), _build_decode_plan(typ, type_table, plan)


def _emit(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Emit *typ*'s schema, ``$ref``-ing it out if it is itself a recursive instantiation."""
    schema_type = type_table.canonical_schema_type(typ)
    if isinstance(schema_type, (RecordType, EnumType)) and schema_type in plan.hoisted:
        return {"$ref": f"#/$defs/{plan.keys[schema_type]}"}
    return _emit_body(schema_type, type_table, plan)


def _emit_body(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Emit *typ*'s own schema body, never ``$ref``-ing *typ* itself.

    Used both for an ordinary (non-recursive) type and for a recursive
    instantiation's own ``"$defs"`` entry — nested fields still route through
    :func:`_emit`, so a recursive instantiation's OWN fields are ``$ref``'d
    exactly like any other occurrence.
    """
    if isinstance(typ, TextType):
        return {"type": "string"}
    if isinstance(typ, IntType):
        return {"type": "integer"}
    if isinstance(typ, DecimalType):
        return {"type": "number"}
    if isinstance(typ, BoolType):
        return {"type": "boolean"}
    if isinstance(typ, JsonType):
        # Permissive: accepts any JSON value.
        return {}
    if isinstance(typ, ArrayType):
        return {"type": "array", "items": _emit(typ.elem, type_table, plan)}
    if isinstance(typ, DictType):
        return {"type": "object", "additionalProperties": _emit(typ.value, type_table, plan)}
    if isinstance(typ, RecordType):
        return _record_schema(typ, type_table, plan)
    if isinstance(typ, EnumType):
        return _enum_schema(typ, type_table, plan)
    if isinstance(typ, ExceptionType):
        raise TypeError(
            f"ExceptionType {typ.name!r} has no JSON Schema; exceptions are not "
            "wire-serialised by the JSON codec."
        )
    if isinstance(typ, UnitType):
        raise TypeError("UnitType has no JSON Schema; unit is not wire-serialised.")
    if isinstance(typ, FunctionType):
        raise TypeError("FunctionType has no JSON Schema; function values are not wire-serialised.")
    if isinstance(typ, BottomType):
        raise TypeError("BottomType has no JSON Schema; bottom type is not wire-serialised.")
    if isinstance(typ, TypeVarType):
        raise TypeError("TypeVarType has no JSON Schema; type variables are not wire-serialised.")
    if isinstance(typ, InferenceVarType):
        raise TypeError("InferenceVarType is internal and has no JSON Schema.")
    assert_never(typ)  # pragma: no cover


def _record_schema(typ: RecordType, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive the JSON Schema for a record type."""
    fields = type_table.record_fields(typ)
    properties: dict[str, object] = {
        field_name: _emit(field_type, type_table, plan) for field_name, field_type in fields.items()
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(fields.keys()),
        "properties": properties,
    }


def _enum_schema(typ: EnumType, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive the JSON Schema for an enum type.

    Each variant becomes a ``oneOf`` alternative.  The ``"$case"`` property
    is a ``const`` string that identifies the selected variant; payload fields
    follow alongside it.
    """
    variant_schemas: list[object] = []
    for variant_name, member in type_table.enum_member_names(typ).items():
        variant_fields = type_table.record_fields(member)
        required: list[str] = ["$case"]
        properties: dict[str, object] = {
            "$case": {"const": variant_name},
        }
        for field_name, field_type in variant_fields.items():
            properties[field_name] = _emit(field_type, type_table, plan)
            required.append(field_name)
        variant_schemas.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
        )
    return {"oneOf": variant_schemas}


# ---------------------------------------------------------------------------
# Recursion planning: the concrete instantiation graph, its SCCs, and the
# deterministic $defs key scheme.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SchemaPlan:
    """Which concrete instantiations reachable from one root get their own ``$defs`` entry.

    ``hoisted`` — every instantiation emitted once into ``$defs`` and referenced
    by ``$ref``/``RefDecode``/``RefEncode`` everywhere it occurs.  An
    instantiation is hoisted when it is recursive FOR THIS ROOT (in a
    non-trivial strongly-connected component of the root's own instantiation
    graph, or with a self-loop) — where a reference is the only way to emit it
    at all — or when it occurs more than once, where sharing one body keeps the
    emitted document proportional to the number of distinct instantiations
    instead of the number of paths that reach them.  ``order`` — those same
    instantiations in first-encounter (breadth-first) order, the order ``$defs``
    entries are built in and the order key collisions are resolved in.
    ``keys`` — each hoisted instantiation's ``$defs`` key (see
    :func:`_assign_defs_keys`).
    """

    hoisted: frozenset[Instantiation]
    order: tuple[Instantiation, ...]
    keys: dict[Instantiation, str] = field(default_factory=dict)


def _plan_schema(typ: Type, type_table: TypeTable) -> _SchemaPlan:
    """Build the recursion plan for *typ*: its instantiation graph, SCCs, and $defs keys."""
    return _plan_types((typ,), type_table)


def _plan_types(types: Iterable[Type], type_table: TypeTable) -> _SchemaPlan:
    """Build one shared recursion plan spanning every type in *types*.

    Multi-root generalization of a single-type plan: the extern boundary plans
    all of an ``extern def``'s parameter types and its result type together, so
    a recursive instantiation shared across them is assigned a single
    ``$defs``/``defs`` key (and one shared body) rather than one per occurrence.
    """
    order, adjacency, occurrences = _build_instantiation_plan(types, type_table)
    if not adjacency:
        return _SchemaPlan(hoisted=frozenset(), order=())
    components = sccs(adjacency, key=_instantiation_sort_key)
    hoisted: set[Instantiation] = {handle for handle, count in occurrences.items() if count > 1}
    for component in components:
        if len(component) > 1:
            hoisted.update(component)
        elif component[0] in adjacency[component[0]]:
            hoisted.add(component[0])
    hoisted_order = tuple(handle for handle in order if handle in hoisted)
    return _SchemaPlan(
        hoisted=frozenset(hoisted),
        order=hoisted_order,
        keys=_assign_defs_keys(hoisted_order, type_table),
    )


def _build_instantiation_plan(
    roots: Iterable[Type], type_table: TypeTable
) -> tuple[
    tuple[Instantiation, ...],
    dict[Instantiation, frozenset[Instantiation]],
    dict[Instantiation, int],
]:
    """Breadth-first expand the concrete record/enum instantiation graph reachable from *roots*.

    Nodes are concrete ``RecordType``/``EnumType``/``ExceptionType`` handles (memoized on handle
    equality); an edge from a node to another is a nominal handle occurring
    anywhere in the node's OWN substituted fields/variants (including nested
    under ``array``/``dict``, or in another reference's own type arguments —
    :func:`~agm.agl.semantics.analyses.nominal_references` finds both).
    Returns ``(order, adjacency, occurrences)``: *order* is first-encounter
    (BFS) order, used for deterministic ``$defs`` key assignment; *adjacency*
    maps each reached handle to its direct neighbours, ready for
    :func:`~agm.util.graph.sccs`; *occurrences* counts how many times each
    handle is referenced across the roots and every expanded node's own
    fields, WITH multiplicity — two fields of the same type are two
    occurrences, which ``adjacency`` deliberately collapses into one edge.
    Seeding from several *roots* in order keeps the first-encounter order
    deterministic across the whole set.
    """
    order: list[Instantiation] = []
    adjacency: dict[Instantiation, frozenset[Instantiation]] = {}
    occurrences: dict[Instantiation, int] = {}
    seen: set[Instantiation] = set()
    root_refs = [
        ref
        for root in roots
        for ref in type_table.schema_relevant_nominal_references(root)
        if isinstance(ref, (RecordType, EnumType, ExceptionType))
    ]
    for ref in root_refs:
        occurrences[ref] = occurrences.get(ref, 0) + 1
    queue: deque[Instantiation] = deque(root_refs)
    while queue:
        handle = queue.popleft()
        if handle in seen:
            continue
        seen.add(handle)
        order.append(handle)
        references = _direct_references(handle, type_table)
        for ref in references:
            occurrences[ref] = occurrences.get(ref, 0) + 1
        neighbours = frozenset(references)
        adjacency[handle] = neighbours
        # Sorted, never frozenset-iteration order: the frozenset's iteration
        # order depends on Python's per-process string-hash randomization, and
        # this order drives both the $defs dict insertion order and the
        # numeric-suffix collision tiebreak in _assign_defs_keys, so it must be
        # deterministic (mirrors the sorted-extension discipline
        # TypeTable.first_infinite_declaration already uses).
        queue.extend(sorted((n for n in neighbours if n not in seen), key=_instantiation_sort_key))
    return tuple(order), adjacency, occurrences


def _direct_references(handle: Instantiation, type_table: TypeTable) -> tuple[Instantiation, ...]:
    """Return every concrete instantiation named in *handle*'s own fields/variants, with repeats.

    Multiplicity is preserved: the caller collapses these into an edge set for
    the SCC pass and counts them for the occurrence tally.
    """
    if isinstance(handle, RecordType):
        field_types: list[Type] = list(type_table.record_fields(handle).values())
    elif isinstance(handle, EnumType):
        field_types = [
            ftype
            for member in type_table.enum_members(handle)
            for ftype in type_table.record_fields(member).values()
        ]
    else:
        field_types = list(type_table.exception_fields(handle).values())
    return tuple(
        ref
        for ftype in field_types
        for ref in type_table.schema_relevant_nominal_references(ftype)
        if isinstance(ref, (RecordType, EnumType, ExceptionType))
    )


def _instantiation_sort_key(handle: Instantiation) -> tuple[object, ...]:
    """Deterministic sort key for :func:`~agm.util.graph.sccs` — never Python object identity.

    Ends in ``decl_id``: two distinct declarations can share every other
    component (a REPL redeclaration mints a fresh identity for the same name
    path), and without a final tiebreak their relative order would fall back
    to a frozenset's hash-randomized iteration order — nondeterministic
    across process runs, unlike every other component here.
    """
    type_args = handle.type_args if isinstance(handle, (RecordType, EnumType)) else ()
    return (
        handle.module_id.segments,
        handle.scope_path,
        handle.name,
        tuple(repr(arg) for arg in type_args),
        handle.decl_id,
    )


# JSON-Schema-safe `$defs` key characters: letters, digits, ``_``, ``.``,
# ``-``.  Any run of other characters (brackets, commas, spaces, the ``::``
# module-qualifier separator) becomes a single ``_`` separator.
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _bare_display(handle: Instantiation, type_table: TypeTable | None = None) -> str:
    """*handle*'s display form WITHOUT its module qualifier (bare name[, args])."""
    type_args: tuple[Type, ...] = ()
    if isinstance(handle, (RecordType, EnumType)):
        type_args = (
            type_table.schema_relevant_type_args(handle)
            if type_table is not None
            else handle.type_args
        )
    if type_args:
        args = ", ".join(repr(arg) for arg in type_args)
        return f"{'::'.join((*handle.scope_path, handle.name))}[{args}]"
    return "::".join((*handle.scope_path, handle.name))


def _sanitize_key(raw: str) -> str:
    return _UNSAFE_KEY_CHARS.sub("_", raw).strip("_")


def _assign_defs_keys(
    order: tuple[Instantiation, ...], type_table: TypeTable | None = None
) -> dict[Instantiation, str]:
    """Assign each recursive instantiation in *order* a deterministic, collision-free ``$defs`` key.

    Base key: the sanitized bare display form (``Tree``, ``Tree_int``).  If
    two DISTINCT instantiations sanitize to the same bare key, BOTH are
    promoted to their module-qualified form (``mod/sub.Tree``) to disambiguate
    — cross-module same-name collision, the common case.  Any collision
    still remaining after that (e.g. two same-module instantiations whose
    argument lists sanitize identically) is broken by a numeric suffix in
    first-encounter order, guaranteeing a collision-free result.
    """
    bare = {handle: _sanitize_key(_bare_display(handle, type_table)) for handle in order}
    bare_counts: dict[str, int] = {}
    for key in bare.values():
        bare_counts[key] = bare_counts.get(key, 0) + 1
    candidates: dict[Instantiation, str] = {}
    for handle in order:
        if bare_counts[bare[handle]] > 1 and not handle.module_id.is_entry:
            qualified = f"{handle.module_id.path_str()}.{_bare_display(handle, type_table)}"
            candidates[handle] = _sanitize_key(qualified)
        else:
            candidates[handle] = bare[handle]
    used: set[str] = set()
    assigned: dict[Instantiation, str] = {}
    for handle in order:
        key = candidates[handle]
        if key in used:
            suffix = 2
            while f"{key}_{suffix}" in used:
                suffix += 1
            key = f"{key}_{suffix}"
        used.add(key)
        assigned[handle] = key
    return assigned


def build_decode_schema(typ: Type, type_table: TypeTable) -> DecodePlan:
    """Compile a checker ``Type`` into a typeless ``DecodePlan``.

    Mirrors :func:`derive_schema`'s recursion handling exactly: the SAME
    recursion plan (:func:`_plan_schema`) drives both, so a recursive
    instantiation's ``DecodePlan.defs`` key matches its JSON Schema ``$defs``
    key one-to-one, and every occurrence of it — including the root itself,
    if recursive — becomes a ``RefDecode(key)`` instead of being inlined.  A
    non-recursive *typ* gets an empty ``defs`` and a ``root`` identical to
    what a plain (non-plan-aware) recursive walk would have produced, so
    non-recursive decode output is unchanged.

    *type_table* resolves record/enum field and variant shapes.

    :raises TypeError: if *typ* has no finite JSON schema at all (see
        ``TypeTable.has_finite_schema``); callers are expected to reject such
        types at the use site before calling this function.
    """
    _require_finite_schema(typ, type_table, "build a decode schema")
    plan = _plan_schema(typ, type_table)
    return _build_decode_plan(typ, type_table, plan)


def _build_decode_plan(typ: Type, type_table: TypeTable, plan: "_SchemaPlan") -> DecodePlan:
    """Build *typ*'s ``DecodePlan`` (root + ``$defs`` entries) from an already-built plan."""
    # One memo per plan, as in :func:`build_encode_plan`: a type reachable by
    # several paths is emitted once and shared, keeping the plan proportional to
    # the number of distinct instantiations rather than to the number of paths.
    memo: dict[Type, DecodeSchema] = {}
    root = _emit_decode(typ, type_table, plan, memo)
    defs = tuple(
        (plan.keys[handle], _emit_decode_body(handle, type_table, plan, memo))
        for handle in plan.order
    )
    return DecodePlan(root=root, defs=defs)


def _emit_decode(
    typ: Type, type_table: TypeTable, plan: "_SchemaPlan", memo: dict[Type, DecodeSchema]
) -> DecodeSchema:
    """Emit *typ*'s decode schema, ``RefDecode``-ing it out if it is a recursive instantiation."""
    schema_type = type_table.canonical_schema_type(typ)
    cached = memo.get(schema_type)
    if cached is not None:
        return cached
    if isinstance(schema_type, (RecordType, EnumType)) and schema_type in plan.hoisted:
        emitted: DecodeSchema = RefDecode(plan.keys[schema_type])
    else:
        emitted = _emit_decode_body(schema_type, type_table, plan, memo)
    memo[schema_type] = emitted
    return emitted


def _emit_decode_body(
    typ: Type, type_table: TypeTable, plan: "_SchemaPlan", memo: dict[Type, DecodeSchema]
) -> DecodeSchema:
    """Emit *typ*'s own decode schema body, never ``RefDecode``-ing *typ* itself.

    Used both for an ordinary (non-recursive) type and for a recursive
    instantiation's own ``defs`` entry — nested fields still route through
    :func:`_emit_decode`, so a recursive instantiation's OWN fields are
    ``RefDecode``'d exactly like any other occurrence.
    """
    if isinstance(typ, TextType):
        return ScalarDecode(ScalarKind.TEXT)
    if isinstance(typ, IntType):
        return ScalarDecode(ScalarKind.INT)
    if isinstance(typ, DecimalType):
        return ScalarDecode(ScalarKind.DECIMAL)
    if isinstance(typ, BoolType):
        return ScalarDecode(ScalarKind.BOOL)
    if isinstance(typ, JsonType):
        return ScalarDecode(ScalarKind.JSON)
    if isinstance(typ, ArrayType):
        return ArrayDecode(_emit_decode(typ.elem, type_table, plan, memo))
    if isinstance(typ, DictType):
        return DictDecode(_emit_decode(typ.value, type_table, plan, memo))
    if isinstance(typ, RecordType):
        fields = type_table.record_fields(typ)
        return RecordDecode(
            nominal=NominalId(typ.decl_id),
            display_name="::".join((*typ.scope_path, typ.name)),
            fields=tuple(
                (fname, _emit_decode(ftype, type_table, plan, memo))
                for fname, ftype in fields.items()
            ),
        )
    if isinstance(typ, EnumType):
        members = type_table.enum_member_names(typ)
        return EnumDecode(
            nominal=NominalId(typ.decl_id),
            display_name="::".join((*typ.scope_path, typ.name)),
            variants=tuple(
                VariantDecode(
                    name=vname,
                    nominal=NominalId(member.decl_id),
                    display_name="::".join((*member.scope_path, member.name)),
                    fields=tuple(
                        (fname, _emit_decode(ftype, type_table, plan, memo))
                        for fname, ftype in vfields.items()
                    ),
                )
                for vname, member in members.items()
                for vfields in (type_table.record_fields(member),)
            ),
        )
    # Non-data targets (unit/function/exception/bottom/typevar) are not
    # decodable from JSON and are rejected by the checker before lowering.
    raise AssertionError(  # pragma: no cover
        f"build_decode_schema: undecodable type {typ!r}"
    )


def build_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a checker ``Type`` into a typeless static JSON encode plan.

    The plan follows the same concrete-instantiation recursion graph as decode
    planning, but is independent of JSON Schema emission. It deliberately
    accepts exceptions because ``as json`` may serialize their fields even
    though exceptions are not JSON decode targets.
    """
    _require_finite_schema(typ, type_table, "build a JSON encode plan")
    plan = _plan_schema(typ, type_table)
    # One memo per plan: a type reachable by several paths is emitted once and
    # its encoder shared, so plan size follows the number of distinct
    # instantiations rather than the number of paths through the type graph.
    memo: dict[Type, EncodeSchema] = {}
    return EncodePlan(
        root=_emit_encode(typ, type_table, plan, memo),
        definitions=tuple(
            EncodeDefinition(
                plan.keys[handle], 0, _emit_encode_body(handle, type_table, plan, memo)
            )
            for handle in plan.order
        ),
    )


def _template_key(nominal: NominalId) -> str:
    """Key a declaration template's definition in a growing plan.

    A growing plan has no JSON Schema counterpart, so — unlike the ``$defs``
    keys of a finite plan (:func:`_assign_defs_keys`) — this key is internal
    and need only be deterministic and collision-free per declaration, which
    the declaration identity already is.
    """
    return f"n{nominal.value}"


def build_dynamic_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a finite generic-template plan for a growing JSON source.

    Concrete instantiations such as ``Perfect[Pair[T, T]]`` grow without a
    finite closure, but their declaration templates are finite. Every
    definition is one declaration's template, keyed by its identity and taking
    one parameter per declared type parameter; each reference retains its
    static slot choices and binds those parameters at the point of use, so the
    runtime never guesses enum membership from a record's nominal identity.
    """
    definitions: dict[NominalId, EncodeDefinition] = {}

    def emit(current: Type, parameters: dict[str, int]) -> EncodeSchema:
        if isinstance(current, (TextType, IntType, DecimalType, BoolType, JsonType)):
            return ScalarEncode()
        if isinstance(current, ArrayType):
            return ArrayEncode(emit(current.elem, parameters))
        if isinstance(current, DictType):
            return DictEncode(emit(current.value, parameters))
        if isinstance(current, TypeVarType):
            index = parameters.get(current.name)
            if index is None:
                raise AssertionError(f"unbound encode type parameter {current.name!r}")
            return TypeParameterEncode(index)
        if isinstance(current, (RecordType, EnumType, ExceptionType)):
            nominal = NominalId(current.decl_id)
            ensure_definition(current)
            args = current.type_args if isinstance(current, (RecordType, EnumType)) else ()
            return RefEncode(_template_key(nominal), tuple(emit(arg, parameters) for arg in args))
        raise AssertionError(f"build a dynamic JSON encode plan: unencodable type {current!r}")

    def ensure_definition(handle: RecordType | EnumType | ExceptionType) -> None:
        nominal = NominalId(handle.decl_id)
        if nominal in definitions:
            return
        typedef = type_table.get_by_id(handle.decl_id)
        if typedef is None:
            raise AssertionError(f"dynamic encode plan references unknown nominal {nominal!r}")
        if typedef.kind == "exception":
            template: RecordType | EnumType | ExceptionType = typedef.handle()
        else:
            template = typedef.handle(tuple(TypeVarType(name) for name in typedef.type_params))
        parameters = {name: index for index, name in enumerate(typedef.type_params)}
        key = _template_key(nominal)
        # Register first so a recursive template can refer to itself while its
        # body is being compiled; replace the temporary once complete.
        definitions[nominal] = EncodeDefinition(key, len(parameters), ScalarEncode())
        if isinstance(template, RecordType):
            body: EncodeSchema = RecordEncode(
                nominal,
                tuple(
                    (name, emit(field_type, parameters))
                    for name, field_type in type_table.record_fields(template).items()
                ),
            )
        elif isinstance(template, ExceptionType):
            body = ExceptionEncode(
                nominal,
                tuple(
                    (name, emit(field_type, parameters))
                    for name, field_type in type_table.exception_fields(template).items()
                ),
            )
        else:
            body = EnumEncode(
                nominal,
                tuple(
                    VariantEncode(
                        name=member_name,
                        nominal=NominalId(member.decl_id),
                        fields=tuple(
                            (field_name, emit(field_type, parameters))
                            for field_name, field_type in type_table.record_fields(member).items()
                        ),
                    )
                    for member_name, member in type_table.enum_member_names(template).items()
                ),
            )
        definitions[nominal] = EncodeDefinition(key, len(parameters), body)

    root = emit(typ, {})
    return EncodePlan(root=root, definitions=tuple(definitions.values()))


def _emit_encode(
    typ: Type, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, EncodeSchema]
) -> EncodeSchema:
    """Emit *typ*'s encoder, referencing recursive bodies through ``defs``."""
    schema_type = type_table.canonical_schema_type(typ)
    cached = memo.get(schema_type)
    if cached is not None:
        return cached
    if (
        isinstance(schema_type, (RecordType, EnumType, ExceptionType))
        and schema_type in plan.hoisted
    ):
        emitted: EncodeSchema = RefEncode(plan.keys[schema_type])
    else:
        emitted = _emit_encode_body(schema_type, type_table, plan, memo)
    memo[schema_type] = emitted
    return emitted


def _emit_encode_body(
    typ: Type, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, EncodeSchema]
) -> EncodeSchema:
    """Emit a non-reference encoder body for one static type."""
    if isinstance(typ, (TextType, IntType, DecimalType, BoolType, JsonType)):
        return ScalarEncode()
    if isinstance(typ, ArrayType):
        return ArrayEncode(_emit_encode(typ.elem, type_table, plan, memo))
    if isinstance(typ, DictType):
        return DictEncode(_emit_encode(typ.value, type_table, plan, memo))
    if isinstance(typ, RecordType):
        return RecordEncode(
            nominal=NominalId(typ.decl_id),
            fields=tuple(
                (name, _emit_encode(field_type, type_table, plan, memo))
                for name, field_type in type_table.record_fields(typ).items()
            ),
        )
    if isinstance(typ, ExceptionType):
        return ExceptionEncode(
            nominal=NominalId(typ.decl_id),
            fields=tuple(
                (name, _emit_encode(field_type, type_table, plan, memo))
                for name, field_type in type_table.exception_fields(typ).items()
            ),
        )
    if isinstance(typ, EnumType):
        return EnumEncode(
            nominal=NominalId(typ.decl_id),
            variants=tuple(
                VariantEncode(
                    name=name,
                    nominal=NominalId(member.decl_id),
                    fields=tuple(
                        (field_name, _emit_encode(field_type, type_table, plan, memo))
                        for field_name, field_type in type_table.record_fields(member).items()
                    ),
                )
                for name, member in type_table.enum_member_names(typ).items()
            ),
        )
    raise AssertionError(f"build_encode_plan: unencodable type {typ!r}")


def build_param_decoder(typ: Type, type_table: TypeTable) -> ParamDecoder:
    """Compile a checker ``Type`` into the typeless ``ParamDecoder`` used to
    decode one host-supplied entry parameter.

    Single source of the param-decoder shape, shared by the lowerer (which
    embeds it in each ``IrParam.external_decoder``) and the REPL/config path
    (:func:`agm.agl.runtime.params.convert_param_value`).  ``text`` params are
    taken verbatim; every other type round-trips through the canonical JSON
    boundary (``derive_schema`` for validation, ``build_decode_schema`` for the
    typeless decode walk).  *type_table* resolves record/enum shapes.

    :raises TypeError: if *typ* has no wire schema (unit/exception/…);
        :func:`derive_schema` rejects such types.
    """
    schema, decode_plan = derive_schema_and_decode(typ, type_table)
    return ParamDecoder(
        target_type_label=repr(typ),
        json_schema=json.dumps(schema, sort_keys=True),
        decode=decode_plan.root,
        defs=decode_plan.defs,
        text_verbatim=isinstance(typ, TextType),
    )


def build_format_instructions(schema: dict[str, object]) -> str:
    """Build agent instructions embedding the authoritative JSON schema."""
    if not schema:
        return "Return exactly one JSON value.\nDo not include Markdown, prose, or code fences."
    schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
    return (
        "Return exactly one JSON value conforming to the following JSON Schema.\n"
        "Do not include Markdown, prose, or code fences.\n"
        "\n"
        f"```json\n{schema_text}\n```"
    )
