"""Compile-time JSON Schema and decode-schema derivation.

:func:`derive_schema` produces a JSON Schema ``dict[str, object]`` from a
semantic :class:`~agm.agl.semantics.types.Type`.  Every entry point in this
module takes an explicit :class:`~agm.agl.semantics.type_table.TypeTable` and
resolves record/enum field and variant shapes through it
(``table.record_fields``/``table.enum_variants``) rather than through the
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

:func:`build_extern_contract` compiles an extern's checked
``FunctionSignature`` into a typeless
:class:`~agm.agl.ir.contracts.ExternContract` describing the shape of every
value crossing the Python FFI boundary — the argument/return type mapping
mirrors :func:`build_decode_schema`'s recursion, with two boundary-specific
differences: ``unit`` compiles (it crosses as Python ``None``, needed for
extern returns) and type-variable positions compile to a
``BoundarySealVar`` leaf instead of being rejected.

Derivation rules:
- ``text``    → ``{"type": "string"}``
- ``int``     → ``{"type": "integer"}``
- ``decimal`` → ``{"type": "number"}``
- ``bool``    → ``{"type": "boolean"}``
- ``json``    → ``{}``  (permissive — accepts any JSON value)
- ``list[T]`` → ``{"type": "array", "items": <schema for T>}``
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
    BoundaryDict,
    BoundaryEnum,
    BoundaryException,
    BoundaryList,
    BoundaryRecord,
    BoundaryRef,
    BoundaryScalar,
    BoundarySchema,
    BoundarySealVar,
    BoundaryUnit,
    BoundaryVariantShape,
    DecodePlan,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ExternContract,
    ExternParamSchema,
    ListDecode,
    ParamDecoder,
    RecordDecode,
    RefDecode,
    ScalarDecode,
    ScalarKind,
    VariantDecode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.semantics.type_table import TypeTable
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
)
from agm.agl.typecheck.env import FunctionSignature
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
    are expected to reject such types at the use site (agent output target,
    cast target, parameter type — see ``typecheck/checker.py`` and
    ``typecheck/builtins.py``), so reaching this guard is an
    internal-invariant violation, not a normal user-facing error path.
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
    if isinstance(schema_type, (RecordType, EnumType)) and schema_type in plan.recursive:
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
    if isinstance(typ, ListType):
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
    if isinstance(typ, AgentType):
        raise TypeError("AgentType has no JSON Schema; agent values are not wire-serialised.")
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
    for variant_name, variant_fields in type_table.enum_variants(typ).items():
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
    """Which concrete instantiations reachable from one root are recursive, and their keys.

    ``recursive`` — every instantiation that is recursive FOR THIS ROOT (in a
    non-trivial strongly-connected component of the root's own instantiation
    graph, or with a self-loop).  ``order`` — those same instantiations in
    first-encounter (breadth-first) order, the order ``$defs`` entries are
    built in and the order key collisions are resolved in.  ``keys`` — each
    recursive instantiation's ``$defs`` key (see :func:`_assign_defs_keys`).
    """

    recursive: frozenset[Instantiation]
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
    order, adjacency = _build_instantiation_plan(types, type_table)
    if not adjacency:
        return _SchemaPlan(recursive=frozenset(), order=())
    components = sccs(adjacency, key=_instantiation_sort_key)
    recursive: set[Instantiation] = set()
    for component in components:
        if len(component) > 1:
            recursive.update(component)
        elif component[0] in adjacency[component[0]]:
            recursive.add(component[0])
    recursive_order = tuple(handle for handle in order if handle in recursive)
    return _SchemaPlan(
        recursive=frozenset(recursive),
        order=recursive_order,
        keys=_assign_defs_keys(recursive_order, type_table),
    )


def _build_instantiation_plan(
    roots: Iterable[Type], type_table: TypeTable
) -> tuple[tuple[Instantiation, ...], dict[Instantiation, frozenset[Instantiation]]]:
    """Breadth-first expand the concrete record/enum instantiation graph reachable from *roots*.

    Nodes are concrete ``RecordType``/``EnumType``/``ExceptionType`` handles (memoized on handle
    equality); an edge from a node to another is a nominal handle occurring
    anywhere in the node's OWN substituted fields/variants (including nested
    under ``list``/``dict``, or in another reference's own type arguments —
    :func:`~agm.agl.semantics.analyses.nominal_references` finds both).
    Returns ``(order, adjacency)``: *order* is first-encounter (BFS) order,
    used for deterministic ``$defs`` key assignment; *adjacency* maps each
    reached handle to its direct neighbours, ready for
    :func:`~agm.util.graph.sccs`.  Seeding from several *roots* in order keeps
    the first-encounter order deterministic across the whole set.
    """
    order: list[Instantiation] = []
    adjacency: dict[Instantiation, frozenset[Instantiation]] = {}
    seen: set[Instantiation] = set()
    queue: deque[Instantiation] = deque(
        ref
        for root in roots
        for ref in type_table.schema_relevant_nominal_references(root)
        if isinstance(ref, (RecordType, EnumType, ExceptionType))
    )
    while queue:
        handle = queue.popleft()
        if handle in seen:
            continue
        seen.add(handle)
        order.append(handle)
        neighbours = _direct_neighbours(handle, type_table)
        adjacency[handle] = neighbours
        # Sorted, never frozenset-iteration order: the frozenset's iteration
        # order depends on Python's per-process string-hash randomization, and
        # this order drives both the $defs dict insertion order and the
        # numeric-suffix collision tiebreak in _assign_defs_keys, so it must be
        # deterministic (mirrors the sorted-extension discipline
        # TypeTable.first_infinite_declaration already uses).
        queue.extend(sorted((n for n in neighbours if n not in seen), key=_instantiation_sort_key))
    return tuple(order), adjacency


def _direct_neighbours(handle: Instantiation, type_table: TypeTable) -> frozenset[Instantiation]:
    """Return every concrete instantiation named anywhere in *handle*'s own fields/variants."""
    if isinstance(handle, RecordType):
        field_types: list[Type] = list(type_table.record_fields(handle).values())
    elif isinstance(handle, EnumType):
        field_types = [
            ftype
            for vfields in type_table.enum_variants(handle).values()
            for ftype in vfields.values()
        ]
    else:
        field_types = list(type_table.exception_fields(handle).values())
    return frozenset(
        ref
        for ftype in field_types
        for ref in type_table.schema_relevant_nominal_references(ftype)
        if isinstance(ref, (RecordType, EnumType, ExceptionType))
    )


def _instantiation_sort_key(handle: Instantiation) -> tuple[object, ...]:
    """Deterministic sort key for :func:`~agm.util.graph.sccs` — never Python object identity."""
    type_args = handle.type_args if isinstance(handle, (RecordType, EnumType)) else ()
    return (handle.module_id.segments, handle.name, tuple(repr(arg) for arg in type_args))


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
        return f"{handle.name}[{args}]"
    return handle.name


def _sanitize_key(raw: str) -> str:
    return _UNSAFE_KEY_CHARS.sub("_", raw).strip("_")


def _assign_defs_keys(
    order: tuple[Instantiation, ...], type_table: TypeTable | None = None
) -> dict[Instantiation, str]:
    """Assign each recursive instantiation in *order* a deterministic, collision-free ``$defs`` key.

    Base key: the sanitized bare display form (``Tree``, ``Tree_int``).  If
    two DISTINCT instantiations sanitize to the same bare key, BOTH are
    promoted to their module-qualified form (``mod.sub.Tree``) to disambiguate
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
            qualified = f"{handle.module_id.dotted()}.{_bare_display(handle, type_table)}"
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
    root = _emit_decode(typ, type_table, plan)
    defs = tuple(
        (plan.keys[handle], _emit_decode_body(handle, type_table, plan)) for handle in plan.order
    )
    return DecodePlan(root=root, defs=defs)


def _emit_decode(typ: Type, type_table: TypeTable, plan: "_SchemaPlan") -> DecodeSchema:
    """Emit *typ*'s decode schema, ``RefDecode``-ing it out if it is a recursive instantiation."""
    schema_type = type_table.canonical_schema_type(typ)
    if isinstance(schema_type, (RecordType, EnumType)) and schema_type in plan.recursive:
        return RefDecode(plan.keys[schema_type])
    return _emit_decode_body(schema_type, type_table, plan)


def _emit_decode_body(typ: Type, type_table: TypeTable, plan: "_SchemaPlan") -> DecodeSchema:
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
    if isinstance(typ, ListType):
        return ListDecode(_emit_decode(typ.elem, type_table, plan))
    if isinstance(typ, DictType):
        return DictDecode(_emit_decode(typ.value, type_table, plan))
    if isinstance(typ, RecordType):
        fields = type_table.record_fields(typ)
        return RecordDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            fields=tuple(
                (fname, _emit_decode(ftype, type_table, plan)) for fname, ftype in fields.items()
            ),
        )
    if isinstance(typ, EnumType):
        variants = type_table.enum_variants(typ)
        return EnumDecode(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            variants=tuple(
                VariantDecode(
                    name=vname,
                    fields=tuple(
                        (fname, _emit_decode(ftype, type_table, plan))
                        for fname, ftype in vfields.items()
                    ),
                )
                for vname, vfields in variants.items()
            ),
        )
    # Non-data targets (unit/agent/function/exception/bottom/typevar) are not
    # decodable from JSON and are rejected by the checker before lowering.
    raise AssertionError(  # pragma: no cover
        f"build_decode_schema: undecodable type {typ!r}"
    )


def build_param_decoder(typ: Type, type_table: TypeTable) -> ParamDecoder:
    """Compile a checker ``Type`` into the typeless ``ParamDecoder`` used to
    decode one host-supplied entry parameter.

    Single source of the param-decoder shape, shared by the lowerer (which
    embeds it in each ``IrParam.external_decoder``) and the REPL/config path
    (:func:`agm.agl.runtime.params.convert_param_value`).  ``text`` params are
    taken verbatim; every other type round-trips through the canonical JSON
    boundary (``derive_schema`` for validation, ``build_decode_schema`` for the
    typeless decode walk).  *type_table* resolves record/enum shapes.

    :raises TypeError: if *typ* has no wire schema (unit/agent/exception/…);
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


def build_extern_contract(sig: FunctionSignature, type_table: TypeTable) -> ExternContract:
    """Compile a checked extern's ``FunctionSignature`` into a typeless ``ExternContract``.

    Walks every parameter type and the result type with :func:`_emit_boundary`,
    recursing through already-instantiated generic nominals exactly as
    :func:`build_decode_schema` does (checker types carry substituted field/variant
    types at the use site, so no separate substitution step is needed here).
    All parameter types and the result type share ONE recursion plan (the same
    :func:`_plan_types` machinery that backs the JSON-schema/decode derivations),
    so a recursive instantiation crosses the boundary as a finite graph of
    ``BoundaryRef`` leaves resolving into :attr:`ExternContract.defs`.

    :raises TypeError: if a function or agent type occurs anywhere in the
        signature, or if any parameter/result type has no finite schema; the
        checker statically rejects both at the extern use site, so these are
        unreachable from source and only exercised by direct invocation.
    """
    boundary_types = (*(param.type for param in sig.params), sig.result)
    for typ in boundary_types:
        _require_finite_schema(typ, type_table, "build an extern contract")
    plan = _plan_types(boundary_types, type_table)
    params = tuple(
        ExternParamSchema(schema=_emit_boundary(param.type, type_table, plan))
        for param in sig.params
    )
    defs = tuple(
        (plan.keys[handle], _emit_boundary_body(handle, type_table, plan)) for handle in plan.order
    )
    return ExternContract(
        params=params,
        result=_emit_boundary(sig.result, type_table, plan),
        type_params=sig.type_params,
        defs=defs,
    )


def _emit_boundary(typ: Type, type_table: TypeTable, plan: "_SchemaPlan") -> BoundarySchema:
    """Emit *typ*'s boundary schema, ``BoundaryRef``-ing it out if it is recursive.

    Mirrors :func:`_emit_decode`: a recursive record/enum/exception instantiation becomes
    a ``BoundaryRef`` into the shared ``defs`` table; everything else is emitted
    inline by :func:`_emit_boundary_body`.
    """
    schema_type = type_table.canonical_schema_type(typ)
    if (
        isinstance(schema_type, (RecordType, EnumType, ExceptionType))
        and schema_type in plan.recursive
    ):
        return BoundaryRef(plan.keys[schema_type])
    return _emit_boundary_body(schema_type, type_table, plan)


def _emit_boundary_body(typ: Type, type_table: TypeTable, plan: "_SchemaPlan") -> BoundarySchema:
    """Emit *typ*'s own boundary schema body, never ``BoundaryRef``-ing *typ* itself.

    Mirrors :func:`build_decode_schema`'s recursion over data types, plus two
    boundary-specific leaves: ``unit`` (crosses as ``None``) and ``TypeVarType``
    (crosses as a sealed opaque handle).  Nested fields route back through
    :func:`_emit_boundary`, so a recursive instantiation's OWN fields are
    ``BoundaryRef``'d exactly like any other occurrence.  Used both for an
    ordinary (non-recursive) type and for a recursive instantiation's own
    ``defs`` entry.
    """
    if isinstance(typ, TextType):
        return BoundaryScalar(ScalarKind.TEXT)
    if isinstance(typ, IntType):
        return BoundaryScalar(ScalarKind.INT)
    if isinstance(typ, DecimalType):
        return BoundaryScalar(ScalarKind.DECIMAL)
    if isinstance(typ, BoolType):
        return BoundaryScalar(ScalarKind.BOOL)
    if isinstance(typ, JsonType):
        return BoundaryScalar(ScalarKind.JSON)
    if isinstance(typ, UnitType):
        return BoundaryUnit()
    if isinstance(typ, ListType):
        return BoundaryList(_emit_boundary(typ.elem, type_table, plan))
    if isinstance(typ, DictType):
        return BoundaryDict(_emit_boundary(typ.value, type_table, plan))
    if isinstance(typ, RecordType):
        return BoundaryRecord(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            fields=tuple(
                (fname, _emit_boundary(ftype, type_table, plan))
                for fname, ftype in type_table.record_fields(typ).items()
            ),
        )
    if isinstance(typ, EnumType):
        return BoundaryEnum(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            variants=tuple(
                BoundaryVariantShape(
                    name=vname,
                    fields=tuple(
                        (fname, _emit_boundary(ftype, type_table, plan))
                        for fname, ftype in vfields.items()
                    ),
                )
                for vname, vfields in type_table.enum_variants(typ).items()
            ),
        )
    if isinstance(typ, ExceptionType):
        # Built-in exceptions resolve under PRELUDE_ID and user exceptions under
        # their declaring module, exactly as the lowerer keys exception nominals
        # (``NominalId(typ.module_id, typ.name)``).
        return BoundaryException(
            nominal=NominalId(typ.module_id, typ.name),
            display_name=typ.name,
            fields=tuple(
                (fname, _emit_boundary(ftype, type_table, plan))
                for fname, ftype in type_table.exception_fields(typ).items()
            ),
        )
    if isinstance(typ, TypeVarType):
        return BoundarySealVar(typ.name)
    if isinstance(typ, InferenceVarType):
        raise TypeError("InferenceVarType is internal and cannot cross the extern boundary.")
    if isinstance(typ, AgentType):
        raise TypeError("AgentType cannot cross the extern boundary; banned in extern signatures.")
    if isinstance(typ, FunctionType):
        raise TypeError(
            "FunctionType cannot cross the extern boundary; banned in extern signatures."
        )
    if isinstance(typ, BottomType):  # pragma: no cover
        # Never assignable to a declared param/result type; unreachable from a
        # checked FunctionSignature.
        raise TypeError("BottomType cannot cross the extern boundary.")
    assert_never(typ)  # pragma: no cover
