"""Compile-time JSON Schema, decode-schema, and encode-plan derivation.

:func:`derive_schema_and_decode` produces a JSON Schema ``dict[str, object]``
and a typeless :class:`~agm.agl.ir.contracts.DecodePlan` from a semantic
:class:`~agm.agl.semantics.types.Type`. Every entry point in this
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

The decode plan (a :class:`~agm.agl.ir.contracts.DecodeSchema` root plus its
``$defs`` table) is used by the IR evaluator to reconstruct typed ``Value``
objects from validated JSON without holding checker ``Type`` references.

:func:`derive_schema_decode_and_tree` also describes a type-directed extern's
target as a typeless :class:`~agm.agl.ir.contracts.TypeTree` from the same
recursion plan, keyed and ordered like the decode plan's ``$defs``.

Derivation rules:
- ``text``    → ``{"type": "string"}``
- ``int``     → ``{"type": "integer"}``
- ``decimal`` → ``{"type": "number"}``
- ``bool``    → ``{"type": "boolean"}``
- ``json``    → ``{}``  (permissive — accepts any JSON value)
- ``array[T]`` → ``{"type": "array", "items": <schema for T>}``
- ``dict[text, V]`` → ``{"type": "object", "additionalProperties": <schema for V>}``
- ``record``  → object schema with ``additionalProperties: false``, ``required``,
                and per-field ``properties``, keyed by each field's effective
                JSON name (``@json-name`` ?? ``@name`` ?? declared). A field
                whose declaration carries a constant default is dropped from
                ``required``.
- ``enum``    → ``{"oneOf": [...]}`` — one variant schema per variant, each an
                object with the member's ``@doc`` as ``description`` when
                present, a ``"$case"`` const property (the member's effective
                JSON tag), and any payload fields, JSON-keyed the same way.

Recursive types: both derivations expand the
concrete *instantiation graph* reachable from *typ* (nodes are concrete
``RecordType``/``EnumType`` handles, edges are the nominal handles occurring
in a node's own substituted fields/variants, memoized on handle equality) and
find its strongly-connected components — computed ONCE per call as a shared
``_SchemaPlan`` (see ``_plan_schema``) that drives both derivations. An instantiation is
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
from collections.abc import Callable, Iterable
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
    FieldDecode,
    FieldEncode,
    ParamDecoder,
    RecordDecode,
    RecordEncode,
    RefDecode,
    RefEncode,
    ScalarDecode,
    ScalarEncode,
    ScalarKind,
    TypeNode,
    TypeNodeField,
    TypeNodeKind,
    TypeNodeRef,
    TypeParameterEncode,
    TypeTree,
    TypeTreeEntry,
    VariantDecode,
    VariantEncode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.external_names import NO_EXTERNAL_NAME
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
    is_standard_agent_enum,
)
from agm.util.graph import sccs

# A concrete nominal instantiation — a graph node in the instantiation graph
# below. Record/enum equality includes type_args; exceptions are non-generic.
Instantiation = RecordType | EnumType | ExceptionType


def _member_tag(member: RecordType, name: str, type_table: TypeTable) -> str:
    """An enum member's effective JSON ``$case`` tag (``@json-name`` ?? ``@name`` ?? declared)."""
    return type_table.external_name(member).json(name)


def _emit_field_decodes(
    handle: RecordType,
    type_table: TypeTable,
    emit_field: "Callable[[Type], DecodeSchema]",
) -> tuple[FieldDecode, ...]:
    """Build one record/member's field decoders via *emit_field*, JSON-keyed, zoned, and aliased.

    ``default_index`` comes straight off ``type_table.field_has_default``: the
    field's position when the field carries a declared default, else ``None``.
    """
    zones = dict(type_table.field_kinds(handle))
    externals = type_table.field_external_names(handle)
    defaults = dict(type_table.field_has_default(handle))
    return tuple(
        FieldDecode(
            name,
            json_name,
            emit_field(ftype),
            zone=zones[name],
            alias=externals.get(name, NO_EXTERNAL_NAME).alias(name),
            default_index=index if defaults[name] else None,
        )
        for index, (name, json_name, ftype) in enumerate(type_table.json_fields(handle))
    )


def _emit_field_encodes(
    handle: RecordType | ExceptionType,
    type_table: TypeTable,
    emit_field: "Callable[[Type], EncodeSchema]",
) -> tuple[FieldEncode, ...]:
    """Build one record/exception's field encoders via *emit_field*, JSON-keyed."""
    return tuple(
        FieldEncode(name, json_name, emit_field(ftype))
        for name, json_name, ftype in type_table.json_fields(handle)
    )


def _require_finite_schema(typ: Type, type_table: TypeTable, action: str) -> None:
    """Raise ``TypeError`` if *typ*'s reachable instantiation closure is infinite.

    Guard for :func:`derive_schema_and_decode`: a type whose recursive instantiations
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
            "derive_schema_and_decode."
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
    """Derive the JSON Schema and the decode plan for *typ* from one shared recursion plan.

    The schema is a valid JSON Schema object; ``Decimal`` and ``int`` values
    round-trip through its validation. *type_table* resolves record/enum field
    and variant shapes. Recursive instantiations reachable from *typ* (see the
    module docstring) are emitted once under a top-level ``"$defs"`` object and
    referenced via ``{"$ref": "#/$defs/<key>"}``, with the decode plan's
    ``defs`` keyed identically; a non-recursive *typ* gets neither.

    A field whose declaration carries a constant default is dropped from its
    record/variant schema's ``required`` list and marked via
    ``FieldDecode.default_index`` in the decode plan; its value is resolved
    only at decode time (see ``runtime.convert.decode_value``'s
    ``default_resolver``).

    :raises TypeError: if *typ* is an ``ExceptionType`` (exceptions are not
        wire-serialised), or if *typ* has no finite JSON schema (see
        ``TypeTable.has_finite_schema``).
    """
    plan = _wire_plan(typ, type_table)
    return _emit_schema_with_plan(typ, type_table, plan), _build_decode_plan(typ, type_table, plan)


def derive_schema_decode_and_tree(
    typ: Type, type_table: TypeTable
) -> tuple[dict[str, object], DecodePlan, TypeTree]:
    """Derive *typ*'s JSON Schema, decode plan, and ``TypeTree`` from one shared recursion plan.

    See :func:`derive_schema_and_decode`; the tree's ``defs`` are keyed and
    ordered like the decode plan's.
    """
    plan = _wire_plan(typ, type_table)
    return (
        _emit_schema_with_plan(typ, type_table, plan),
        _build_decode_plan(typ, type_table, plan),
        _build_type_tree(typ, type_table, plan),
    )


def _wire_plan(typ: Type, type_table: TypeTable) -> _SchemaPlan:
    """Build *typ*'s recursion plan, rejecting a type with no JSON Schema up front."""
    if isinstance(typ, ExceptionType):
        raise TypeError(
            f"ExceptionType {typ.name!r} has no JSON Schema; exceptions are not "
            "wire-serialised by the JSON codec."
        )
    _require_finite_schema(typ, type_table, "derive a JSON Schema/decode plan")
    return _plan_schema(typ, type_table)


def _emit(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Emit *typ*'s schema, ``$ref``-ing it out if it is itself a recursive instantiation."""
    return _emit_planned(
        typ,
        type_table,
        plan,
        plan.schemas,
        lambda body_type: _emit_body(body_type, type_table, plan),
        lambda key: {"$ref": f"#/$defs/{key}"},
    )


def _emit_body(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Emit *typ*'s own schema body once per plan, never ``$ref``-ing *typ* itself.

    Used both for an ordinary (non-recursive) type and for a recursive
    instantiation's own ``"$defs"`` entry — nested fields still route through
    :func:`_emit`, so a recursive instantiation's OWN fields are ``$ref``'d
    exactly like any other occurrence.
    """
    body = plan.bodies.get(typ)
    if body is None:
        body = plan.bodies[typ] = _derive_body(typ, type_table, plan)
    return body


def _derive_body(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive *typ*'s own schema body; see :func:`_emit_body`."""
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


def _record_properties(
    handle: RecordType, type_table: TypeTable, plan: _SchemaPlan
) -> tuple[list[str], dict[str, object]]:
    """Build one record/member's ``required``+``properties`` pair, JSON-keyed.

    A field whose declaration carries a constant default is dropped from
    ``required``; every other field stays required. Shared by
    :func:`_record_schema` and :func:`_enum_schema`, which each own their
    respective object's own envelope (``$case`` for an enum variant).
    """
    defaults = dict(type_table.field_has_default(handle))
    required: list[str] = []
    properties: dict[str, object] = {}
    for name, json_name, field_type in type_table.json_fields(handle):
        properties[json_name] = _emit(field_type, type_table, plan)
        if not defaults[name]:
            required.append(json_name)
    return required, properties


def _record_schema(typ: RecordType, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive the JSON Schema for a record type, keyed by its effective JSON field names."""
    required, properties = _record_properties(typ, type_table, plan)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _enum_schema(typ: EnumType, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive the JSON Schema for an enum type.

    Each variant becomes a ``oneOf`` alternative.  The ``"$case"`` property is
    a ``const`` string that identifies the selected variant — the member's
    effective JSON tag; payload fields follow, keyed by their effective JSON
    names, each dropped from ``required`` on the same terms as a record field
    (see :func:`_record_properties`). A documented member carries its
    ``@doc`` prose as the alternative's ``description`` annotation.
    """
    return {
        "oneOf": [
            _variant_schema(member, variant_name, type_table, plan)
            for variant_name, member in type_table.enum_member_names(typ).items()
        ]
    }


def _variant_schema(
    member: RecordType, variant_name: str, type_table: TypeTable, plan: _SchemaPlan
) -> dict[str, object]:
    """Emit one enum member's ``oneOf`` alternative once per plan."""
    cached = plan.variants.get(member)
    if cached is not None:
        return cached
    required_fields, field_properties = _record_properties(member, type_table, plan)
    required = ["$case", *required_fields]
    properties = {
        "$case": {"const": _member_tag(member, variant_name, type_table)},
        **field_properties,
    }
    variant_schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    doc = type_table.declaration_doc(member)
    if doc is not None:
        variant_schema["description"] = doc
    plan.variants[member] = variant_schema
    return variant_schema


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
    :func:`_assign_defs_keys`).  ``schemas``/``bodies``/``variants`` memoize
    the JSON Schema fragments emitted under this plan: each canonical type's
    occurrence and own body, and each enum member's alternative.
    """

    hoisted: frozenset[Instantiation]
    order: tuple[Instantiation, ...]
    keys: dict[Instantiation, str] = field(default_factory=dict)
    schemas: dict[Type, dict[str, object]] = field(default_factory=dict)
    bodies: dict[Type, dict[str, object]] = field(default_factory=dict)
    variants: dict[RecordType, dict[str, object]] = field(default_factory=dict)


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


def _emit_planned[S](
    typ: Type,
    type_table: TypeTable,
    plan: "_SchemaPlan",
    memo: dict[Type, S],
    body: "Callable[[Type], S]",
    ref: "Callable[[str], S]",
) -> S:
    """Emit *typ* once per plan: a *ref* to its ``$defs`` key when hoisted, else its *body*."""
    schema_type = type_table.canonical_schema_type(typ)
    cached = memo.get(schema_type)
    if cached is not None:
        return cached
    emitted = ref(plan.keys[schema_type]) if schema_type in plan.hoisted else body(schema_type)
    memo[schema_type] = emitted
    return emitted


def _emit_decode(
    typ: Type, type_table: TypeTable, plan: "_SchemaPlan", memo: dict[Type, DecodeSchema]
) -> DecodeSchema:
    """Emit *typ*'s decode schema, ``RefDecode``-ing it out if it is a recursive instantiation."""
    return _emit_planned(
        typ,
        type_table,
        plan,
        memo,
        lambda body_type: _emit_decode_body(body_type, type_table, plan, memo),
        RefDecode,
    )


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
        return RecordDecode(
            nominal=NominalId(typ.decl_id),
            display_name="::".join((*typ.scope_path, typ.name)),
            fields=_emit_field_decodes(
                typ, type_table, lambda ftype: _emit_decode(ftype, type_table, plan, memo)
            ),
            name=typ.name,
            alias=type_table.external_name(typ).alias(typ.name),
        )
    if isinstance(typ, EnumType):
        members = type_table.enum_member_names(typ)
        return EnumDecode(
            nominal=NominalId(typ.decl_id),
            display_name="::".join((*typ.scope_path, typ.name)),
            variants=tuple(
                VariantDecode(
                    name=vname,
                    json_name=_member_tag(member, vname, type_table),
                    nominal=NominalId(member.decl_id),
                    display_name="::".join((*member.scope_path, member.name)),
                    fields=_emit_field_decodes(
                        member,
                        type_table,
                        lambda ftype: _emit_decode(ftype, type_table, plan, memo),
                    ),
                    alias=type_table.external_name(member).alias(vname),
                )
                for vname, member in members.items()
            ),
            name=typ.name,
            host_agent=is_standard_agent_enum(typ),
        )
    # Non-data targets (unit/function/exception/bottom/typevar) are not
    # decodable from JSON and are rejected by the checker before lowering.
    raise AssertionError(  # pragma: no cover
        f"undecodable type {typ!r}"
    )


def _build_type_tree(typ: Type, type_table: TypeTable, plan: _SchemaPlan) -> TypeTree:
    """Build *typ*'s ``TypeTree`` (root + ``defs``) from an already-built plan."""
    memo: dict[Type, TypeTreeEntry] = {}
    return TypeTree(
        root=_emit_tree(typ, type_table, plan, memo),
        defs=tuple(
            (plan.keys[handle], _emit_tree_body(handle, type_table, plan, memo))
            for handle in plan.order
        ),
    )


_SCALAR_NODE_KINDS: dict[type[Type], TypeNodeKind] = {
    TextType: TypeNodeKind.TEXT,
    IntType: TypeNodeKind.INT,
    DecimalType: TypeNodeKind.DECIMAL,
    BoolType: TypeNodeKind.BOOL,
    JsonType: TypeNodeKind.JSON,
}


def _emit_tree(
    typ: Type, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, TypeTreeEntry]
) -> TypeTreeEntry:
    """Emit *typ*'s tree entry, ``TypeNodeRef``-ing it out if it is hoisted."""
    return _emit_planned(
        typ,
        type_table,
        plan,
        memo,
        lambda body_type: _emit_tree_body(body_type, type_table, plan, memo),
        TypeNodeRef,
    )


def _emit_tree_body(
    typ: Type, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, TypeTreeEntry]
) -> TypeNode:
    """Emit *typ*'s own tree node, never ``TypeNodeRef``-ing *typ* itself."""
    schema = json.dumps(_emit_body(typ, type_table, plan))
    label = repr(typ)
    if isinstance(typ, ArrayType):
        return TypeNode(
            TypeNodeKind.ARRAY, label, schema, items=_emit_tree(typ.elem, type_table, plan, memo)
        )
    if isinstance(typ, DictType):
        return TypeNode(
            TypeNodeKind.DICT, label, schema, values=_emit_tree(typ.value, type_table, plan, memo)
        )
    if isinstance(typ, RecordType):
        return _nominal_tree_node(
            TypeNodeKind.RECORD,
            typ,
            schema,
            type_table,
            fields=_emit_tree_fields(typ, type_table, plan, memo),
        )
    if isinstance(typ, EnumType):
        return _nominal_tree_node(
            TypeNodeKind.ENUM,
            typ,
            schema,
            type_table,
            members=tuple(
                (
                    _member_tag(member, name, type_table),
                    _nominal_tree_node(
                        TypeNodeKind.MEMBER,
                        member,
                        json.dumps(_variant_schema(member, name, type_table, plan)),
                        type_table,
                        fields=_emit_tree_fields(member, type_table, plan, memo),
                    ),
                )
                for name, member in type_table.enum_member_names(typ).items()
            ),
        )
    # Every non-scalar data type is handled above; the schema emitter has
    # already rejected every non-data type.
    return TypeNode(_SCALAR_NODE_KINDS[type(typ)], label, schema)


def _nominal_tree_node(
    kind: TypeNodeKind,
    handle: RecordType | EnumType,
    schema: str,
    type_table: TypeTable,
    *,
    fields: tuple[TypeNodeField, ...] = (),
    members: tuple[tuple[str, TypeNode], ...] = (),
) -> TypeNode:
    """Build a record, enum, or member node carrying its declaration's doc and identity."""
    return TypeNode(
        kind,
        repr(handle),
        schema,
        doc=type_table.declaration_doc(handle),
        nominal=NominalId(handle.decl_id),
        fields=fields,
        members=members,
    )


def _emit_tree_fields(
    handle: RecordType, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, TypeTreeEntry]
) -> tuple[TypeNodeField, ...]:
    """Build one record/member's field entries, JSON-keyed and documented."""
    docs = type_table.field_docs(handle)
    return tuple(
        TypeNodeField(name, json_name, docs.get(name), _emit_tree(ftype, type_table, plan, memo))
        for name, json_name, ftype in type_table.json_fields(handle)
    )


def build_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a checker ``Type`` into the typeless JSON encode plan for it.

    Both plan shapes are the same closed node family; only how their
    definitions are keyed and parameterized differs, so the choice between
    them is made here rather than by every caller.
    """
    if type_table.has_finite_schema(typ):
        return _build_finite_encode_plan(typ, type_table)
    return _build_template_encode_plan(typ, type_table)


def _build_finite_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a plan over the concrete instantiations a finite source reaches.

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


def _build_template_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a finite declaration-template plan for a growing JSON source.

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
                nominal, _emit_field_encodes(template, type_table, lambda ft: emit(ft, parameters))
            )
        elif isinstance(template, ExceptionType):
            body = ExceptionEncode(
                nominal, _emit_field_encodes(template, type_table, lambda ft: emit(ft, parameters))
            )
        else:
            body = EnumEncode(
                nominal,
                tuple(
                    VariantEncode(
                        name=member_name,
                        json_name=_member_tag(member, member_name, type_table),
                        nominal=NominalId(member.decl_id),
                        fields=_emit_field_encodes(
                            member, type_table, lambda ft: emit(ft, parameters)
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
    return _emit_planned(
        typ,
        type_table,
        plan,
        memo,
        lambda body_type: _emit_encode_body(body_type, type_table, plan, memo),
        RefEncode,
    )


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
            fields=_emit_field_encodes(
                typ, type_table, lambda ftype: _emit_encode(ftype, type_table, plan, memo)
            ),
        )
    if isinstance(typ, ExceptionType):
        return ExceptionEncode(
            nominal=NominalId(typ.decl_id),
            fields=_emit_field_encodes(
                typ, type_table, lambda ftype: _emit_encode(ftype, type_table, plan, memo)
            ),
        )
    if isinstance(typ, EnumType):
        return EnumEncode(
            nominal=NominalId(typ.decl_id),
            variants=tuple(
                VariantEncode(
                    name=name,
                    json_name=_member_tag(member, name, type_table),
                    nominal=NominalId(member.decl_id),
                    fields=_emit_field_encodes(
                        member,
                        type_table,
                        lambda ftype: _emit_encode(ftype, type_table, plan, memo),
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
    embeds it in each ``IrProgramParam.external_decoder``)
    and the host engine-config decode path
    (:func:`agm.agl.runtime.engine_config.convert_host_value`).  A textual raw
    value is read through the shared host-text dispatch
    (``runtime.value_decode.host_text_to_json``), which derives ``text``-verbatim
    and standard-``Agent`` handling from ``decode`` itself; every other type
    round-trips through the canonical JSON boundary
    (:func:`derive_schema_and_decode`: schema validation, then the typeless
    decode walk).
    *type_table* resolves record/enum shapes.

    :raises TypeError: if *typ* has no wire schema (unit/exception/…);
        :func:`derive_schema_and_decode` rejects such types.
    """
    schema, decode_plan = derive_schema_and_decode(typ, type_table)
    return ParamDecoder(
        target_type_label=repr(typ),
        json_schema=json.dumps(schema, sort_keys=True),
        decode=decode_plan.root,
        defs=decode_plan.defs,
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
