"""Normalize checked AgL patterns into canonical pattern-matrix rows."""

from __future__ import annotations

import decimal
import weakref
from dataclasses import replace
from typing import assert_never, cast

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    BottomType,
    CheckedType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax.nodes import (
    AsPattern,
    BoolLit,
    Case,
    ConstructorPattern,
    DecimalLit,
    IntLit,
    LiteralPattern,
    NullLit,
    Pattern,
    StringLit,
    VarPattern,
    WildcardPattern,
)
from agm.agl.typecheck.env import CheckedModule
from agm.util.decimal import exact_decimal

from .model import (
    BinderProvenance,
    BoolConstructor,
    CaseSite,
    ClosedSignature,
    Constructor,
    ConstructorCell,
    ConstructorField,
    LiteralConstructor,
    LiteralKind,
    MatchCaseContext,
    MatrixRow,
    NominalConstructor,
    NormalizedMatchSite,
    Occurrence,
    OccurrenceId,
    OmittedFieldProvenance,
    OpenSignature,
    PatternCell,
    RootOccurrenceProvenance,
    Signature,
    SourceAction,
    SourcePatternProvenance,
    WildcardCell,
)

CheckedPatternOwner = CheckedModule


class MatchCompileInvariantError(RuntimeError):
    """A checked-program invariant required by match compilation was violated."""


def resolve_bare_enum_constructors(
    checked: CheckedPatternOwner,
) -> frozenset[tuple[ModuleId, str, str]]:
    """Collect enum constructors whose unqualified call forms are visible.

    The witness renderer may use an explicit call form for field-bearing
    variants, so its visibility set is broader than the nullary-only bare-name
    pattern rule. Ordinary value bindings do not hide these pattern forms.

    A candidate only qualifies when its owner path is a registered enum's own
    declaration path, so the key's middle component really is an enum name. A
    record declared in a named scope never qualifies: its owner path is its
    declaration's enclosing scope, even when that scope shares a name with an
    unrelated enum.
    """
    enum_paths = {
        (typedef.module_id, (*typedef.scope_path, typedef.name))
        for typedef in checked.type_env.type_table.entries()
        if typedef.kind == "enum"
    }
    return frozenset(
        (candidate.owner_module_id, candidate.owner_path[-1], candidate.owner_name)
        for candidates in checked.resolved.constructor_candidates.values()
        for candidate in candidates
        if (candidate.owner_module_id, candidate.owner_path) in enum_paths
    )


def enum_constructor(enum_type: EnumType, variant: str, table: TypeTable) -> NominalConstructor:
    return record_constructor(table.enum_member_names(enum_type)[variant], table)


def record_constructor(record_type: RecordType, table: TypeTable) -> NominalConstructor:
    fields = table.record_fields(record_type)
    return NominalConstructor(
        record_type=record_type,
        fields=tuple(ConstructorField(name, field_type) for name, field_type in fields.items()),
    )


def constructor_inhabits_type(
    constructor: Constructor, subject_type: Type, table: TypeTable
) -> bool:
    """Return whether a constructor denotes any runtime value of ``subject_type``.

    This dispatch is deliberately total over the checked type union.  In
    particular, runtime numeric equality permits an integral decimal pattern
    to match an integer, but no integer value can equal a fractional or
    non-finite decimal.  AgL decimal values are finite exact decimals.
    """
    match cast(CheckedType, subject_type):
        case BoolType():
            return isinstance(constructor, BoolConstructor)
        case EnumType() as enum_type:
            members = table.enum_members(enum_type)
            return (
                isinstance(constructor, NominalConstructor) and constructor.record_type in members
            )
        case RecordType() as record_type:
            return (
                isinstance(constructor, NominalConstructor)
                and constructor.record_type == record_type
            )
        case IntType():
            return (
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.NUMERIC
                and isinstance(constructor.value, decimal.Decimal)
                and constructor.value.is_finite()
                and constructor.value == constructor.value.to_integral_value()
            )
        case DecimalType():
            return (
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.NUMERIC
                and isinstance(constructor.value, decimal.Decimal)
                and constructor.value.is_finite()
            )
        case TextType():
            return (
                isinstance(constructor, LiteralConstructor) and constructor.kind is LiteralKind.TEXT
            )
        case JsonType():
            return (
                isinstance(constructor, LiteralConstructor) and constructor.kind is LiteralKind.NULL
            )
        case (
            TypeVarType()
            | ArrayType()
            | DictType()
            | ExceptionType()
            | UnitType()
            | FunctionType()
            | BottomType()
        ):
            return False
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def pattern_cell_inhabits_type(cell: PatternCell, subject_type: Type, table: TypeTable) -> bool:
    """Return whether a canonical cell can match a value of ``subject_type``."""
    if isinstance(subject_type, BottomType):
        return False
    if isinstance(cell, WildcardCell):
        return True
    if not constructor_inhabits_type(cell.constructor, subject_type, table):
        return False
    if isinstance(cell.constructor, NominalConstructor):
        return all(
            pattern_cell_inhabits_type(argument, field.type, table)
            for field, argument in zip(cell.constructor.fields, cell.arguments, strict=True)
        )
    return True


_NOMINAL_SIGNATURES: weakref.WeakKeyDictionary[
    TypeTable, dict[tuple[EnumType | RecordType, TypeDef], ClosedSignature]
] = weakref.WeakKeyDictionary()


def _build_enum_signature(enum_type: EnumType, table: TypeTable) -> ClosedSignature:
    variant_names = tuple(table.enum_member_names(enum_type))
    return ClosedSignature(
        tuple(enum_constructor(enum_type, name, table) for name in variant_names)
    )


def _build_record_signature(record_type: RecordType, table: TypeTable) -> ClosedSignature:
    return ClosedSignature((record_constructor(record_type, table),))


def _nominal_signature(nominal_type: EnumType | RecordType, table: TypeTable) -> ClosedSignature:
    """Return a declaration-sensitive closed signature for a nominal type.

    Memoized per ``(handle, its declaration)``: pairing the handle with the
    ``TypeDef`` it NAMES — looked up by the handle's own declaration identity,
    never by its bare name — is what makes a redeclaration of exactly that
    declaration invalidate the entry, without an unrelated declaration
    sharing the name affecting it either way. A checked program's nominal
    types always name a registered declaration.
    """

    def build() -> ClosedSignature:
        if isinstance(nominal_type, EnumType):
            return _build_enum_signature(nominal_type, table)
        return _build_record_signature(nominal_type, table)

    typedef = table.typedef_of(nominal_type.decl_id)
    cache = _NOMINAL_SIGNATURES.get(table)
    if cache is None:
        cache = {}
        _NOMINAL_SIGNATURES[table] = cache
    key = (nominal_type, typedef)
    signature = cache.get(key)
    if signature is None:
        signature = build()
        cache[key] = signature
    return signature


def signature_for_type(subject_type: Type, table: TypeTable) -> Signature:
    """Return the complete constructor signature for every current semantic type.

    The explicit closed dispatch is intentional: adding a semantic ``Type``
    without classifying its matching domain is a compiler error, not an implicit
    fallback to an open domain.
    """
    match cast(CheckedType, subject_type):
        case BoolType():
            return ClosedSignature((BoolConstructor(False), BoolConstructor(True)))
        case EnumType() | RecordType() as nominal_type:
            return _nominal_signature(nominal_type, table)
        case BottomType():
            return ClosedSignature(())
        case (
            TextType()
            | JsonType()
            | IntType()
            | DecimalType()
            | TypeVarType()
            | ArrayType()
            | DictType()
            | ExceptionType()
            | UnitType()
            | FunctionType()
        ):
            return OpenSignature()
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _canonical_literal(pattern: LiteralPattern) -> Constructor:
    """Return the constructor of a literal pattern, which checking matched to its occurrence."""
    literal = pattern.literal
    match literal:
        case BoolLit():
            return BoolConstructor(literal.value)
        case IntLit():
            return LiteralConstructor(LiteralKind.NUMERIC, exact_decimal(literal.value))
        case DecimalLit():
            return LiteralConstructor(LiteralKind.NUMERIC, literal.value)
        case StringLit():
            return LiteralConstructor(LiteralKind.TEXT, literal.value)
        case NullLit():
            return LiteralConstructor(LiteralKind.NULL, None)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _add_as_binder(cell: PatternCell, binder: BinderProvenance) -> PatternCell:
    """Attach an as-pattern binder to the occurrence represented by *cell*."""
    return replace(cell, binders=(*cell.binders, binder))


def _nominal_constructor(
    subject_type: EnumType | RecordType, owner: int, table: TypeTable
) -> NominalConstructor:
    """Return the constructor of the record declared as *owner* within *subject_type*."""
    if isinstance(subject_type, EnumType):
        return record_constructor(
            cast(RecordType, table.enum_member_by_decl(subject_type, owner)), table
        )
    return record_constructor(subject_type, table)


def normalize_pattern(
    pattern: Pattern,
    subject_type: Type,
    checked: CheckedPatternOwner,
) -> PatternCell:
    """Normalize one checked pattern against its checked occurrence type."""
    provenance = SourcePatternProvenance(pattern.node_id, pattern.span)
    match pattern:
        case WildcardPattern():
            return WildcardCell(provenance=provenance)
        case AsPattern(pattern=inner, node_id=node_id, name=name):
            return _add_as_binder(
                normalize_pattern(inner, subject_type, checked),
                BinderProvenance(node_id=node_id, name=name, span=pattern.span),
            )
        case VarPattern(node_id=node_id, name=name):
            constructor_ref = checked.pattern_classifications[node_id]
            if constructor_ref is None:
                return WildcardCell(
                    provenance=provenance,
                    binders=(BinderProvenance(node_id=node_id, name=name, span=pattern.span),),
                )
            constructor = _nominal_constructor(
                cast("EnumType | RecordType", subject_type),
                constructor_ref.owner_decl_node_id,
                checked.type_env.type_table,
            )
            return ConstructorCell(constructor, (), provenance)
        case LiteralPattern():
            return ConstructorCell(
                constructor=_canonical_literal(pattern),
                arguments=(),
                provenance=provenance,
            )
        case ConstructorPattern():
            # Checker-published nominal identity; normalization does not
            # re-select the constructor from scope candidates.
            nominal_constructor = _nominal_constructor(
                cast("EnumType | RecordType", subject_type),
                checked.pattern_constructor_owners[pattern.node_id].value,
                checked.type_env.type_table,
            )
            supplied = dict(checked.argument_bindings.constructor_patterns[pattern.node_id])
            arguments = tuple(
                normalize_pattern(supplied[field.name], field.type, checked)
                if field.name in supplied
                else WildcardCell(
                    provenance=OmittedFieldProvenance(field_name=field.name, span=pattern.span)
                )
                for field in nominal_constructor.fields
            )
            return ConstructorCell(nominal_constructor, arguments, provenance)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def match_case_context(checked: CheckedPatternOwner) -> MatchCaseContext:
    """Resolve the checked qualification metadata shared by one owner's match sites."""
    return MatchCaseContext(
        module_id=checked.module_id,
        enum_owner_forms=checked.type_env.enum_owner_forms(),
        blocked_enum_variants=checked.type_env.blocked_enum_variants(),
        bare_enum_constructors=resolve_bare_enum_constructors(checked),
        owner_program=checked.resolved.program,
    )


def normalize_case(
    case: Case,
    checked: CheckedPatternOwner,
    *,
    case_context: MatchCaseContext | None = None,
) -> NormalizedMatchSite:
    """Normalize one checked source case into a source-priority one-column matrix.

    An owner-wide caller passes *case_context* so all of its match sites share
    one resolution of the checked qualification metadata; direct callers let it
    default to resolving from *checked*.
    """
    subject_type = checked.node_types[case.subject.node_id]
    root = Occurrence(
        id=OccurrenceId(0),
        creation_order=0,
        type=subject_type,
        provenance=RootOccurrenceProvenance(
            site_node_id=case.node_id,
            span=case.subject.span,
        ),
    )
    rows: list[MatrixRow] = []
    for index, branch in enumerate(case.branches):
        cell = normalize_pattern(branch.pattern, subject_type, checked)
        if pattern_cell_inhabits_type(cell, subject_type, checked.type_env.type_table):
            rows.append(
                MatrixRow(
                    cells=(cell,),
                    action_id=branch.node_id,
                    source_index=index,
                    source_pattern_id=branch.pattern.node_id,
                )
            )
    actions = tuple(
        SourceAction(
            action_id=branch.node_id,
            source_index=index,
            pattern_span=branch.pattern.span,
        )
        for index, branch in enumerate(case.branches)
    )
    return NormalizedMatchSite(
        site_node_id=case.node_id,
        source=CaseSite(actions=actions),
        span=case.span,
        root=root,
        occurrences=(root,),
        rows=tuple(rows),
        type_table=checked.type_env.type_table,
        case_context=case_context if case_context is not None else match_case_context(checked),
    )


__all__ = [
    "CheckedPatternOwner",
    "MatchCompileInvariantError",
    "constructor_inhabits_type",
    "enum_constructor",
    "match_case_context",
    "normalize_case",
    "normalize_pattern",
    "pattern_cell_inhabits_type",
    "record_constructor",
    "resolve_bare_enum_constructors",
    "signature_for_type",
]
