"""Coercion compiler for the AgL lowering phase.

``compile_coercion(source, target, type_table)`` is the ONLY place that reads
checker ``Type`` objects to produce a ``Coercion`` descriptor.  Once this
function returns, the coercion is fully pre-resolved; the evaluator only
switches on the returned ``Coercion`` union and never sniffs value types at
runtime.

Ordering follows the contract exactly (mirrors legacy eval/interpreter._coerce):
  1. equal types → None (identity)
  2. target is JsonType and source is not JsonType → ToJson
  3. target is DecimalType and source is IntType → IntToDecimal
  4. both ListType → recurse on elem; MapList if child is not None
  5. both DictType → recurse on value; MapDictValues if child is not None
  6. both RecordType → per shared field; MapRecordFields if any
  7. both EnumType → per variant/field; MapEnumFields if any
  8. otherwise → None (identity / opaque / no implicit coercion)

TypeVarType sources or targets, equal types, and the json→json identity all
return None.  Record/enum field and variant shapes are resolved through the
shared ``TypeTable`` (``table.record_fields``/``table.enum_variants``) rather
than the handle's own embedded maps.

The equal-types check runs FIRST, before the record/enum field walk, so that
a recursive declaration's self-reference — a field typed exactly as its own
enclosing record/enum (or the same generic instantiation) — short-circuits
instead of re-expanding the same fields forever: two ``RecordType``/
``EnumType`` handles are equal iff their ``(module_id, name, type_args)``
identity matches, which is exactly the condition under which the shared
``TypeTable`` substitutes their templates identically, so no coercion is ever
needed between them. For a finite (non-recursive) type this reordering is
unobservable — record/enum handles under the same identity always yield the
same fields — but for a recursive type it is the difference between
terminating and an unbounded ``compile_coercion`` recursion.

Opaqueness generalizes from a bare ``TypeVarType`` to ANY type that still
CONTAINS a free type variable (``contains_type_var``), not just one that IS
one. This matters for a recursive generic nominal parameter: the lowerer
calls a generic function's/constructor's signature parameter types as
declared (a TEMPLATE such as ``Tree[T]``, since substituting a call site's
concrete type arguments into the whole signature is deferred to this
function rather than done again per call), while the argument expression's
own checked type is already concrete (e.g. ``Tree[int]``). Recursing into
``Tree[T]``'s own fields never makes progress — substituting ``type_args =
(T,)`` into ``Tree``'s template is an identity substitution, so the SAME
``(source, target)`` pair reappears every step, an unbounded recursion for a
recursive declaration (list/dict/function templates instead reach a bare
``TypeVarType`` leaf after finitely many steps and stop, so they never
actually needed this rule — but nothing is lost by covering them too).
Treating any such pair as opaque is exactly right, not just crash-avoidance:
nominal types are invariant (see the "Invariance" section of
``docs/agl/reference/generics.md``), so whenever a template still containing
a type variable is accepted at a call site, the checker's assignability
already forced the argument's concrete type to match the template's eventual
substitution exactly — no coercion is ever actually needed there.

Genuinely unequal instantiations of the SAME recursive declaration (e.g.
``Box[int]`` vs. ``Box[decimal]``, both fully concrete, where ``Box[T]`` has
a ``list[Box[T]]`` field) are a further, hypothetical case the two checks
above do not cover, and one investigation did not find reachable through the
checker either: the same invariance argument applies at every nesting depth,
so two syntactically different FULLY CONCRETE handles of the same recursive
declaration are never assignable to each other and this function is never
asked to coerce between them today. ``_compile_coercion`` still tracks
in-flight ``(source, target)`` pairs and raises ``AssertionError`` (an
internal-invariant violation, not a user-facing diagnostic — mirroring
``TypeTable``'s cyclic exception-base-chain guard) if one is ever re-entered,
so a future change that loosens assignability fails loudly with a clear
message instead of a raw Python ``RecursionError``.
"""

from __future__ import annotations

from agm.agl.ir.operations import (
    Coercion,
    IntToDecimal,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
)
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    DecimalType,
    DictType,
    EnumType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    Type,
    contains_type_var,
)

__all__ = ["compile_coercion"]


def compile_coercion(source: Type, target: Type, type_table: TypeTable) -> Coercion | None:
    """Compile an implicit coercion from *source* to *target*.

    Returns a ``Coercion`` descriptor to be wrapped in an ``IrCoerce`` node, or
    ``None`` when no coercion node is needed (identity / opaque / equal types).
    *type_table* resolves record/enum field and variant shapes.

    Ordering (mirrors legacy eval/interpreter._coerce):
      1. equal types → None
      2. target is json and source is not json → ToJson
      3. target is decimal and source is int → IntToDecimal
      4. both list → recurse on elem; MapList if child is not None
      5. both dict → recurse on value; MapDictValues if child is not None
      6. both record → per shared field; MapRecordFields if any
      7. both enum → per variant/field; MapEnumFields if any
      8. otherwise → None

    See the module docstring for why the equality check runs before the
    record/enum field walk (recursive-type termination), and for why an
    unequal same-declaration recursive pair is guarded rather than reachable.
    """
    return _compile_coercion(source, target, type_table, visiting=frozenset())


def _compile_coercion(
    source: Type,
    target: Type,
    type_table: TypeTable,
    *,
    visiting: frozenset[tuple[Type, Type]],
) -> Coercion | None:
    # Opaque: a type still containing a free type variable — anywhere, not
    # just at the top level — stands for a value whose shape isn't pinned
    # down at THIS point (a bare `T`, or a template like `Tree[T]` whose call
    # site hasn't been substituted into the signature). See the module
    # docstring for why no coercion is ever actually needed in that case.
    if contains_type_var(source) or contains_type_var(target):
        return None

    # 1. Equal types: identity, no coercion. Checked before the record/enum
    # field walk below so a recursive type's self-reference (a field typed
    # exactly as the enclosing declaration) terminates immediately instead of
    # re-expanding the same fields forever.
    if source == target:
        return None

    # Defense in depth (see module docstring): an unequal pair of the same
    # recursive declaration is not reachable through the checker today, but
    # if this exact (source, target) pair is already being compiled higher up
    # the call stack, recursing into it again would never terminate. Fail
    # loudly with an internal diagnostic instead of a raw RecursionError.
    pair = (source, target)
    if pair in visiting:
        raise AssertionError(
            "compiler bug: compile_coercion re-entered the same (source, target) "
            f"pair {pair!r} — an unequal instantiation of a recursive nominal "
            "declaration reached the coercion compiler, which the checker's "
            "invariant nominal-type assignability is expected to prevent."
        )
    visiting = visiting | {pair}

    # 6. Both record → per shared field.
    if isinstance(target, RecordType) and isinstance(source, RecordType):
        src_fields = type_table.record_fields(source)
        tgt_fields = type_table.record_fields(target)
        field_ops: list[tuple[str, Coercion]] = []
        for field_name, tgt_field_type in tgt_fields.items():
            src_field_type = src_fields.get(field_name)
            if src_field_type is None:
                continue
            child = _compile_coercion(src_field_type, tgt_field_type, type_table, visiting=visiting)
            if child is not None:
                field_ops.append((field_name, child))
        return MapRecordFields(tuple(field_ops)) if field_ops else None

    # 7. Both enum → per variant, per field.
    if isinstance(target, EnumType) and isinstance(source, EnumType):
        src_variants = type_table.enum_variants(source)
        tgt_variants = type_table.enum_variants(target)
        variant_ops: list[tuple[str, tuple[tuple[str, Coercion], ...]]] = []
        for variant_name, tgt_vfields in tgt_variants.items():
            src_vfields = src_variants.get(variant_name, {})
            field_ops_v: list[tuple[str, Coercion]] = []
            for field_name, tgt_ftype in tgt_vfields.items():
                src_ftype = src_vfields.get(field_name)
                if src_ftype is None:
                    continue
                child = _compile_coercion(src_ftype, tgt_ftype, type_table, visiting=visiting)
                if child is not None:
                    field_ops_v.append((field_name, child))
            if field_ops_v:
                variant_ops.append((variant_name, tuple(field_ops_v)))
        return MapEnumFields(tuple(variant_ops)) if variant_ops else None

    # 2. Target is json and source is not json → wrap in ToJson.
    if isinstance(target, JsonType):
        # source != target so source is not JsonType (equal check above)
        return ToJson()

    # 3. Target is decimal and source is int → widen.
    if isinstance(target, DecimalType) and isinstance(source, IntType):
        return IntToDecimal()

    # 4. Both list → recurse on element type.
    if isinstance(target, ListType) and isinstance(source, ListType):
        child = _compile_coercion(source.elem, target.elem, type_table, visiting=visiting)
        return MapList(child) if child is not None else None

    # 5. Both dict → recurse on value type.
    if isinstance(target, DictType) and isinstance(source, DictType):
        child = _compile_coercion(source.value, target.value, type_table, visiting=visiting)
        return MapDictValues(child) if child is not None else None

    # 8. Otherwise: no implicit coercion.
    return None
