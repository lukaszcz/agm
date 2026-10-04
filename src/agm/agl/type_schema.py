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

:func:`derive_schema_and_tree` describes a type-directed extern's
target as a typeless :class:`~agm.agl.ir.contracts.TypeTree` from the same
recursion plan, keyed and ordered like the decode plan's ``$defs``.

Derivation rules:
- ``text``    → ``{"type": "string"}``
- ``int``     → ``{"type": "integer"}``
- ``decimal`` → ``{"type": "number"}``
- ``bool``    → ``{"type": "boolean"}``
- ``json``    → ``{}``  (permissive — accepts any JSON value)
- ``array[T]`` → ``{"type": "array", "items": <schema for T>}``
- ``dict[K, V]`` → one of three forms chosen by ``K`` (see
  ``ir.contracts.DictKeyForm``): a ``text``-keyed dict is
  ``{"type": "object", "additionalProperties": <schema for V>}``; an
  int/decimal/bool/all-nullary-enum key stringifies onto the same object shape
  with a ``propertyNames`` constraint; every other hashable key (records,
  enums with payload members, ``json``, ``Option[int]``, exceptions) emits
  ``{"type": "array", "items": {"type": "object",
  "properties": {"key": <schema for K>, "value": <schema for V>}, ...}}``.
- ``record``  → object schema with ``additionalProperties: false``, ``required``,
                and per-field ``properties``, keyed by each field's effective
                JSON name (``@json-name`` ?? ``@name`` ?? declared), with each
                field's ``@doc`` as its property's ``description`` when present.
                A field whose declaration carries a constant default is dropped
                from ``required``.
- ``enum``    → ``{"oneOf": [...]}`` — one variant schema per variant, each an
                object with the member's ``@doc`` as ``description`` when
                present, a ``"$case"`` const property (the member's effective
                JSON tag), and any payload fields, JSON-keyed the same way and
                carrying their own ``@doc`` as ``description`` when present.
                A plain enum (every member fieldless) is instead its member's
                tag string: ``{"enum": [tags]}``, or — when any member carries
                a ``@doc`` — ``{"oneOf": [...]}`` of ``{"const": tag}``
                alternatives, each with its member's ``@doc`` as
                ``description``.

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
unchanged from a purely-inlining derivation. A schema is derived only for a
checked wire data type with a finite instantiation closure
(``type_table.has_finite_schema``); the checker rejects every other type at
its use site.
"""

from __future__ import annotations

import json
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

from agm.agl.ir.contracts import (
    ArrayDecode,
    ArrayEncode,
    DecodePlan,
    DecodeSchema,
    DictDecode,
    DictEncode,
    DictKeyForm,
    EncodeDefinition,
    EncodePlan,
    EncodeSchema,
    EnumDecode,
    EnumEncode,
    ExceptionEncode,
    ExceptionFieldEncode,
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
    dict_key_form,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.external_names import NO_EXTERNAL_NAME
from agm.agl.semantics.type_table import TypeTable, is_json_convertible
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    is_standard_agent_enum,
)
from agm.util.graph import sccs

# A concrete nominal instantiation — a graph node in the instantiation graph
# below. Record/enum equality includes type_args. An exception is never a node:
# it encodes by its runtime nominal (``ExceptionEncode``), so a plan never
# descends into its fields.
Instantiation = RecordType | EnumType


def _is_plain_enum(typ: EnumType, type_table: TypeTable) -> bool:
    """Whether every member of *typ* is fieldless (mirrors ``ir.contracts.is_plain_enum``)."""
    return not any(type_table.record_fields(member) for member in type_table.enum_members(typ))


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
    handle: RecordType,
    type_table: TypeTable,
    emit_field: "Callable[[Type], EncodeSchema]",
) -> tuple[FieldEncode, ...]:
    """Build one record/member's field encoders via *emit_field*, JSON-keyed."""
    return tuple(
        FieldEncode(name, json_name, emit_field(ftype))
        for name, json_name, ftype in type_table.json_fields(handle)
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

    """
    plan = _plan_schema(typ, type_table)
    return _emit_schema_with_plan(typ, type_table, plan), _build_decode_plan(typ, type_table, plan)


def derive_schema_and_tree(typ: Type, type_table: TypeTable) -> tuple[dict[str, object], TypeTree]:
    """Derive *typ*'s JSON Schema and ``TypeTree`` from one shared recursion plan.

    See :func:`derive_schema_and_decode`; the tree's ``defs`` are keyed and
    ordered like the schema's ``$defs``.
    """
    plan = _plan_schema(typ, type_table)
    return (
        _emit_schema_with_plan(typ, type_table, plan),
        _build_type_tree(typ, type_table, plan),
    )


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
        return _dict_schema(typ, type_table, plan)
    if isinstance(typ, RecordType):
        return _record_schema(typ, type_table, plan)
    # The checker admits only wire data types; the one left is an enum.
    return _enum_schema(cast(EnumType, typ), type_table, plan)


#: JSON-number grammar an int or decimal key stringifies to: an optional sign, then ``0`` or a
#: non-zero-leading digit run, an optional fractional part, and an optional exponent. One
#: pattern for both — decoding validates against the derived schema, and the decode rule
#: accepts any JSON number text for an int target (e.g. ``"2.0"``), so integrality is enforced
#: by the decoder rather than the schema.
_NUMBER_KEY_PATTERN = r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$"


def _stringified_key_schema(head: EncodeSchema) -> dict[str, object]:
    """``propertyNames`` constraint matching a stringified key's exact wire text.

    *head* is the key's own resolved encode-schema head — the SAME head
    :func:`~agm.agl.ir.contracts.dict_key_form` classified to choose this
    form (see :func:`_dict_schema`) — so a bool/int/decimal key's
    ``ScalarKind`` and an enum key's ``json_name`` per variant are read
    straight off it, never re-derived from *type_table*.
    """
    if isinstance(head, ScalarEncode):
        if head.kind is ScalarKind.BOOL:
            return {"enum": ["true", "false"]}
        return {"pattern": _NUMBER_KEY_PATTERN}
    enum_head = cast(EnumEncode, head)
    return {"enum": [variant.json_name for variant in enum_head.variants]}


def _dict_schema(typ: DictType, type_table: TypeTable, plan: _SchemaPlan) -> dict[str, object]:
    """Derive a dict type's JSON Schema, branching on its key's ``DictKeyForm``."""
    value_schema = _emit(typ.value, type_table, plan)
    key_encode_schema = _emit_encode(typ.key, type_table, plan, plan.encode_heads)
    key_head = _key_head(typ.key, key_encode_schema, type_table, plan, plan.encode_heads)
    form = dict_key_form(key_head)
    if form is DictKeyForm.OBJECT_TEXT:
        return {"type": "object", "additionalProperties": value_schema}
    if form is DictKeyForm.OBJECT_STRINGIFIED:
        return {
            "type": "object",
            "propertyNames": _stringified_key_schema(key_head),
            "additionalProperties": value_schema,
        }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["key", "value"],
            "properties": {"key": _emit(typ.key, type_table, plan), "value": value_schema},
        },
    }


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
    docs = type_table.field_docs(handle)
    required: list[str] = []
    properties: dict[str, object] = {}
    for name, json_name, field_type in type_table.json_fields(handle):
        field_schema = _emit(field_type, type_table, plan)
        doc = docs.get(name)
        if doc is not None:
            field_schema = {**field_schema, "description": doc}
        properties[json_name] = field_schema
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

    A plain enum's value is its member's tag string, so its alternatives are
    ``const`` tags; with no documented member they collapse to one ``enum``
    list.
    """
    members = type_table.enum_member_names(typ)
    if not _is_plain_enum(typ, type_table):
        return {
            "oneOf": [
                _variant_schema(member, variant_name, type_table, plan)
                for variant_name, member in members.items()
            ]
        }
    alternatives = [
        _plain_variant_schema(member, variant_name, type_table)
        for variant_name, member in members.items()
    ]
    if any("description" in alternative for alternative in alternatives):
        return {"oneOf": alternatives}
    return {"enum": [alternative["const"] for alternative in alternatives]}


def _plain_variant_schema(
    member: RecordType, variant_name: str, type_table: TypeTable
) -> dict[str, object]:
    """Emit one plain-enum member's alternative: its ``const`` tag, documented when it has a doc."""
    schema: dict[str, object] = {"const": type_table.member_json_tag(member, variant_name)}
    doc = type_table.declaration_doc(member)
    if doc is not None:
        schema["description"] = doc
    return schema


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
        "$case": {"const": type_table.member_json_tag(member, variant_name)},
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
    ``encode_heads`` memoizes each dict key's own emitted encode schema (see
    :func:`_emit_encode`) — a bare ``RefEncode`` for a hoisted key, else its
    own body — shared across every dict key :func:`_dict_schema` derives
    under this plan. ``hoisted_bodies`` memoizes each hoisted instantiation's
    own fully resolved encode-schema body (see :func:`_hoisted_body`), keyed
    by the canonical instantiation rather than by occurrence: shared by
    :func:`_dict_schema`, ``_build_finite_encode_plan``'s definitions loop,
    and :func:`_emit_encode_body`'s ``DictType`` arm, so a hoisted key's body
    — needed by any of the three to classify its ``DictKeyForm`` — is built
    at most once per plan.
    """

    hoisted: frozenset[Instantiation]
    order: tuple[Instantiation, ...]
    keys: dict[Instantiation, str] = field(default_factory=dict)
    schemas: dict[Type, dict[str, object]] = field(default_factory=dict)
    bodies: dict[Type, dict[str, object]] = field(default_factory=dict)
    variants: dict[RecordType, dict[str, object]] = field(default_factory=dict)
    encode_heads: dict[Type, EncodeSchema] = field(default_factory=dict)
    hoisted_bodies: "dict[Instantiation, EncodeSchema]" = field(default_factory=dict)


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

    Nodes are concrete ``RecordType``/``EnumType`` handles (memoized on handle
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
        if isinstance(ref, (RecordType, EnumType))
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
    else:
        field_types = [
            ftype
            for member in type_table.enum_members(handle)
            for ftype in type_table.record_fields(member).values()
        ]
    return tuple(
        ref
        for ftype in field_types
        for ref in type_table.schema_relevant_nominal_references(ftype)
        if isinstance(ref, (RecordType, EnumType))
    )


def _instantiation_sort_key(handle: Instantiation) -> tuple[object, ...]:
    """Deterministic sort key for :func:`~agm.util.graph.sccs` — never Python object identity.

    Ends in ``decl_id``: two distinct declarations can share every other
    component (a REPL redeclaration mints a fresh identity for the same name
    path), and without a final tiebreak their relative order would fall back
    to a frozenset's hash-randomized iteration order — nondeterministic
    across process runs, unlike every other component here.
    """
    return (
        handle.module_id.segments,
        handle.scope_path,
        handle.name,
        tuple(repr(arg) for arg in handle.type_args),
        handle.decl_id,
    )


# JSON-Schema-safe `$defs` key characters: letters, digits, ``_``, ``.``,
# ``-``.  Any run of other characters (brackets, commas, spaces, the ``::``
# module-qualifier separator) becomes a single ``_`` separator.
_UNSAFE_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _bare_display(handle: Instantiation, type_table: TypeTable | None = None) -> str:
    """*handle*'s display form WITHOUT its module qualifier (bare name[, args])."""
    type_args = (
        type_table.schema_relevant_type_args(handle) if type_table is not None else handle.type_args
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
    if isinstance(typ, (TextType, IntType, DecimalType, BoolType, JsonType)):
        return ScalarDecode(_scalar_kind(typ))
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
    # The checker admits only wire data types; the one left is an enum.
    enum_type = cast(EnumType, typ)
    members = type_table.enum_member_names(enum_type)
    return EnumDecode(
        nominal=NominalId(enum_type.decl_id),
        display_name="::".join((*enum_type.scope_path, enum_type.name)),
        variants=tuple(
            VariantDecode(
                name=vname,
                json_name=type_table.member_json_tag(member, vname),
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
        name=enum_type.name,
        host_agent=is_standard_agent_enum(enum_type),
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

_SCALAR_KINDS: dict[type[Type], ScalarKind] = {
    TextType: ScalarKind.TEXT,
    IntType: ScalarKind.INT,
    DecimalType: ScalarKind.DECIMAL,
    BoolType: ScalarKind.BOOL,
    JsonType: ScalarKind.JSON,
}


def _scalar_kind(typ: Type) -> ScalarKind:
    """Map one of the five scalar/``json`` leaf types to its ``ScalarKind``."""
    return _SCALAR_KINDS[type(typ)]


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
            TypeNodeKind.DICT,
            label,
            schema,
            keys=_emit_tree(typ.key, type_table, plan, memo),
            values=_emit_tree(typ.value, type_table, plan, memo),
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
        plain = _is_plain_enum(typ, type_table)
        return _nominal_tree_node(
            TypeNodeKind.ENUM,
            typ,
            schema,
            type_table,
            members=tuple(
                (
                    type_table.member_json_tag(member, name),
                    _nominal_tree_node(
                        TypeNodeKind.MEMBER,
                        member,
                        json.dumps(
                            _plain_variant_schema(member, name, type_table)
                            if plain
                            else _variant_schema(member, name, type_table, plan)
                        ),
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


def build_exception_field_encodes(
    handle: ExceptionType, type_table: TypeTable
) -> tuple[ExceptionFieldEncode, ...]:
    """Compile one exception's own field encodes, the entries ``ExceptionEncode`` selects.

    Each field's JSON name is its effective external name (``@json-name`` ??
    ``@name`` ?? declared), covering every field so no two can collide. A
    field with no JSON form carries no plan: a cast of it fails, and an
    uncaught-exception report falls back to the value-directed walk (see
    ``runtime.serialize``).
    """
    return tuple(
        ExceptionFieldEncode(
            field_name,
            json_name,
            build_encode_plan(field_type, type_table)
            if is_json_convertible(field_type, type_table)
            else None,
        )
        for field_name, json_name, field_type in type_table.json_fields(handle)
    )


def _build_finite_encode_plan(typ: Type, type_table: TypeTable) -> EncodePlan:
    """Compile a plan over the concrete instantiations a finite source reaches.

    The plan follows the same concrete-instantiation recursion graph as decode
    planning, but is independent of JSON Schema emission. It deliberately
    accepts exceptions because ``as json`` may serialize their fields even
    though exceptions are not JSON decode targets.
    """
    plan = _plan_schema(typ, type_table)
    # One memo per plan: a type reachable by several paths is emitted once and
    # its encoder shared, so plan size follows the number of distinct
    # instantiations rather than the number of paths through the type graph.
    memo: dict[Type, EncodeSchema] = {}
    return EncodePlan(
        root=_emit_encode(typ, type_table, plan, memo),
        definitions=tuple(
            EncodeDefinition(plan.keys[handle], 0, _hoisted_body(handle, type_table, plan, memo))
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
    one parameter per schema-relevant type parameter; each reference retains
    its static slot choices and binds those parameters at the point of use,
    so the runtime never guesses enum membership from a record's nominal
    identity.
    """
    definitions: dict[NominalId, EncodeDefinition] = {}
    in_progress: set[NominalId] = set()

    def emit(current: Type, parameters: dict[str, int]) -> EncodeSchema:
        if isinstance(current, (TextType, IntType, DecimalType, BoolType, JsonType)):
            return ScalarEncode(_scalar_kind(current))
        if isinstance(current, ArrayType):
            return ArrayEncode(emit(current.elem, parameters))
        if isinstance(current, DictType):
            key_schema = emit(current.key, parameters)
            value_schema = emit(current.value, parameters)
            # A dict key must be ``Hashable``, so a key handle's own fields
            # can hold no ``DictType`` anywhere in their structure; building
            # the key's definition below can therefore never re-enter this
            # arm on the SAME handle, so ``ensure_definition`` always runs to
            # completion (never short-circuits on its in-progress guard)
            # before returning here, and this reads the key's resolved body
            # straight out of ``definitions`` rather than re-deriving its
            # shape from declaration data. A key reached only through a
            # PHANTOM type argument (see ``ensure_definition``) never gets
            # here at all: its enclosing argument is dropped before ``emit``
            # is ever called on it.
            key_form = (
                None
                if isinstance(key_schema, TypeParameterEncode)
                else dict_key_form(
                    definitions[NominalId(cast("RecordType | EnumType", current.key).decl_id)].body
                    if isinstance(key_schema, RefEncode)
                    else key_schema
                )
            )
            return DictEncode(key_form, key_schema, value_schema)
        if isinstance(current, TypeVarType):
            return TypeParameterEncode(parameters[current.name])
        if isinstance(current, ExceptionType):
            return ExceptionEncode(NominalId(current.decl_id))
        # The checker admits only encodable types; the ones left are records and enums.
        handle = cast("RecordType | EnumType", current)
        ensure_definition(handle)
        return RefEncode(
            _template_key(NominalId(handle.decl_id)),
            tuple(emit(arg, parameters) for arg in type_table.schema_relevant_type_args(handle)),
        )

    def ensure_definition(handle: RecordType | EnumType) -> None:
        nominal = NominalId(handle.decl_id)
        if nominal in definitions or nominal in in_progress:
            return
        # Marked in-progress (not yet registered) so a recursive template can
        # refer to itself — via the ``RefEncode`` its own ``emit`` call below
        # builds regardless — while its body is still being compiled, without
        # a placeholder body ever entering ``definitions``.
        in_progress.add(nominal)
        typedef = type_table.typedef_of(handle.decl_id)
        template = cast(
            "RecordType | EnumType",
            typedef.handle(tuple(TypeVarType(name) for name in typedef.type_params)),
        )
        # Only the declaration's own SCHEMA-RELEVANT parameters get a slot: a
        # phantom parameter never reaches a field/variant position, so no
        # ``TypeParameterEncode`` in the body below can ever name one, and a
        # reference (see ``emit``'s handle arm) supplies one argument per
        # relevant name, in declaration order.
        relevant = type_table.schema_relevant_params(handle.decl_id)
        parameters = {
            name: index
            for index, name in enumerate(p for p in typedef.type_params if p in relevant)
        }
        key = _template_key(nominal)
        if isinstance(template, RecordType):
            body: EncodeSchema = RecordEncode(
                nominal, _emit_field_encodes(template, type_table, lambda ft: emit(ft, parameters))
            )
        else:
            body = EnumEncode(
                nominal,
                tuple(
                    VariantEncode(
                        name=member_name,
                        json_name=type_table.member_json_tag(member, member_name),
                        nominal=NominalId(member.decl_id),
                        fields=_emit_field_encodes(
                            member, type_table, lambda ft: emit(ft, parameters)
                        ),
                    )
                    for member_name, member in type_table.enum_member_names(template).items()
                ),
            )
        definitions[nominal] = EncodeDefinition(key, len(parameters), body)
        in_progress.discard(nominal)

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


def _hoisted_body(
    handle: "RecordType | EnumType",
    type_table: TypeTable,
    plan: _SchemaPlan,
    memo: dict[Type, EncodeSchema],
) -> EncodeSchema:
    """Return *handle*'s own resolved encode-schema body, built at most once per plan.

    *handle* is always hoisted (a ``RefEncode`` names it), so its body cannot
    itself be reached again from within its own construction — any
    self/mutual reference to it, anywhere in its fields, is emitted as the
    SAME ``RefEncode`` rather than re-descending here (see
    :func:`_emit_encode`) — and building it is therefore safe to do on
    first demand rather than up front. A dict key is ``Hashable``, so its
    body can hold no ``DictType`` field anywhere in its own structure;
    building one handle's body here therefore never re-enters
    ``_hoisted_body`` to resolve some OTHER key's head. Shared by
    :func:`_build_finite_encode_plan`'s definitions loop, :func:`_dict_schema`,
    and this module's own ``DictType`` encode-plan arm (see
    ``_SchemaPlan.hoisted_bodies``), so a key's own body needed to classify
    its :class:`~agm.agl.ir.contracts.DictKeyForm` is built once regardless of
    how many of those three ask for it.
    """
    cached = plan.hoisted_bodies.get(handle)
    if cached is not None:
        return cached
    body = _emit_encode_body(handle, type_table, plan, memo)
    plan.hoisted_bodies[handle] = body
    return body


def _key_head(
    dict_key: Type,
    key_schema: EncodeSchema,
    type_table: TypeTable,
    plan: _SchemaPlan,
    memo: dict[Type, EncodeSchema],
) -> EncodeSchema:
    """Resolve a dict key's already-emitted *key_schema* to its concrete, non-ref head.

    A hoisted key (record/enum) emits as a ``RefEncode``; its body -- the
    same head :func:`~agm.agl.ir.contracts.dict_key_form` classifies -- comes
    from :func:`_hoisted_body` rather than being re-descended.
    """
    return (
        _hoisted_body(cast("RecordType | EnumType", dict_key), type_table, plan, memo)
        if isinstance(key_schema, RefEncode)
        else key_schema
    )


def _emit_encode_body(
    typ: Type, type_table: TypeTable, plan: _SchemaPlan, memo: dict[Type, EncodeSchema]
) -> EncodeSchema:
    """Emit a non-reference encoder body for one static type."""
    if isinstance(typ, (TextType, IntType, DecimalType, BoolType, JsonType)):
        return ScalarEncode(_scalar_kind(typ))
    if isinstance(typ, ArrayType):
        return ArrayEncode(_emit_encode(typ.elem, type_table, plan, memo))
    if isinstance(typ, DictType):
        key_schema = _emit_encode(typ.key, type_table, plan, memo)
        value_schema = _emit_encode(typ.value, type_table, plan, memo)
        key_head = _key_head(typ.key, key_schema, type_table, plan, memo)
        return DictEncode(dict_key_form(key_head), key_schema, value_schema)
    if isinstance(typ, RecordType):
        return RecordEncode(
            nominal=NominalId(typ.decl_id),
            fields=_emit_field_encodes(
                typ, type_table, lambda ftype: _emit_encode(ftype, type_table, plan, memo)
            ),
        )
    if isinstance(typ, ExceptionType):
        return ExceptionEncode(NominalId(typ.decl_id))
    # The checker admits only encodable types; the one left is an enum.
    enum_type = cast(EnumType, typ)
    return EnumEncode(
        nominal=NominalId(enum_type.decl_id),
        variants=tuple(
            VariantEncode(
                name=name,
                json_name=type_table.member_json_tag(member, name),
                nominal=NominalId(member.decl_id),
                fields=_emit_field_encodes(
                    member,
                    type_table,
                    lambda ftype: _emit_encode(ftype, type_table, plan, memo),
                ),
            )
            for name, member in type_table.enum_member_names(enum_type).items()
        ),
    )


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
