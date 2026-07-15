"""Normalize checked AgL patterns into canonical pattern-matrix rows."""

from __future__ import annotations

import decimal
from typing import assert_never

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.symbols import BinderKind, ScopeNode
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
from agm.agl.syntax.nodes import (
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
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.graph import CheckedModule

from .model import (
    BinderProvenance,
    BoolConstructor,
    ClosedSignature,
    Constructor,
    ConstructorCell,
    ConstructorField,
    EnumConstructor,
    LiteralConstructor,
    LiteralKind,
    MatchCaseContext,
    MatrixRow,
    NormalizedCase,
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

CheckedPatternOwner = CheckedProgram | CheckedModule


class MatchCompileInvariantError(RuntimeError):
    """A checked-program invariant required by match compilation was violated."""


def _bare_enum_constructors(
    scope: ScopeNode,
    checked: CheckedPatternOwner,
) -> frozenset[tuple[ModuleId, str, str]]:
    """Collect declaration-level bare variants valid in the exact case scope."""
    result: set[tuple[ModuleId, str, str]] = set()
    for candidate in checked.resolved.bare_variant_refs.values():
        binding = scope.lookup(candidate.variant or "")
        if (
            binding is None
            or binding.kind is not BinderKind.constructor_binding
            or candidate.variant is None
        ):
            continue
        result.add((candidate.owner_module_id, candidate.owner_name, candidate.variant))
    return frozenset(result)


def _enum_constructor(
    enum_type: EnumType, variant: str, table: TypeTable
) -> EnumConstructor:
    try:
        variants = table.enum_variants(enum_type)
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError(
            f"cannot resolve enum signature for checked type {enum_type!r}"
        ) from exc
    fields = variants.get(variant)
    if fields is None:
        raise MatchCompileInvariantError(
            f"checked enum pattern names unknown variant {enum_type!r}::{variant}"
        )
    return EnumConstructor(
        enum_type=enum_type,
        variant=variant,
        fields=tuple(ConstructorField(name, field_type) for name, field_type in fields.items()),
    )


def constructor_inhabits_type(constructor: Constructor, subject_type: Type) -> bool:
    """Return whether a constructor denotes any runtime value of ``subject_type``.

    This dispatch is deliberately total over both current closed unions.  In
    particular, runtime numeric equality permits an integral decimal pattern
    to match an integer, but no integer value can equal a fractional or
    non-finite decimal.  AgL decimal values are finite exact decimals.
    """
    match constructor:
        case BoolConstructor() | EnumConstructor() | LiteralConstructor():
            pass
        case _ as unsupported_constructor:
            try:
                assert_never(unsupported_constructor)
            except AssertionError as exc:
                raise MatchCompileInvariantError(
                    f"unsupported constructor {type(unsupported_constructor).__name__}"
                ) from exc

    match subject_type:
        case BoolType():
            return isinstance(constructor, BoolConstructor)
        case EnumType() as enum_type:
            return (
                isinstance(constructor, EnumConstructor)
                and constructor.enum_type == enum_type
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
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.TEXT
            )
        case JsonType():
            return (
                isinstance(constructor, LiteralConstructor)
                and constructor.kind is LiteralKind.NULL
            )
        case InferenceVarType():
            raise MatchCompileInvariantError(
                "flexible inference type escaped checked output"
            )
        case (
            TypeVarType()
            | ListType()
            | DictType()
            | RecordType()
            | ExceptionType()
            | UnitType()
            | AgentType()
            | FunctionType()
            | BottomType()
        ):
            return False
        case _ as unsupported_type:
            try:
                assert_never(unsupported_type)
            except AssertionError as exc:
                raise MatchCompileInvariantError(
                    f"unsupported semantic type {type(unsupported_type).__name__}"
                ) from exc


def pattern_cell_inhabits_type(cell: PatternCell, subject_type: Type) -> bool:
    """Return whether a canonical cell can match a value of ``subject_type``."""
    if isinstance(subject_type, BottomType):
        return False
    if isinstance(cell, WildcardCell):
        return True
    if not constructor_inhabits_type(cell.constructor, subject_type):
        return False
    if isinstance(cell.constructor, EnumConstructor):
        return all(
            pattern_cell_inhabits_type(argument, field.type)
            for field, argument in zip(
                cell.constructor.fields, cell.arguments, strict=True
            )
        )
    return True


def signature_for_type(subject_type: Type, table: TypeTable) -> Signature:
    """Return the complete constructor signature for every current semantic type.

    The explicit closed dispatch is intentional: adding a semantic ``Type``
    without classifying its matching domain is a compiler error, not an implicit
    fallback to an open domain.
    """
    match subject_type:
        case BoolType():
            return ClosedSignature((BoolConstructor(False), BoolConstructor(True)))
        case EnumType() as enum_type:
            try:
                variant_names = tuple(table.enum_variants(enum_type))
            except (KeyError, AssertionError) as exc:
                raise MatchCompileInvariantError(
                    f"cannot resolve enum signature for checked type {enum_type!r}"
                ) from exc
            return ClosedSignature(
                tuple(_enum_constructor(enum_type, name, table) for name in variant_names)
            )
        case InferenceVarType():
            raise MatchCompileInvariantError(
                "flexible inference type escaped checked output"
            )
        case BottomType():
            return ClosedSignature(())
        case (
            TextType()
            | JsonType()
            | IntType()
            | DecimalType()
            | TypeVarType()
            | ListType()
            | DictType()
            | RecordType()
            | ExceptionType()
            | UnitType()
            | AgentType()
            | FunctionType()
        ):
            return OpenSignature()
        case _ as unreachable:
            try:
                assert_never(unreachable)
            except AssertionError as exc:
                raise MatchCompileInvariantError(
                    f"unsupported semantic type {type(unreachable).__name__}"
                ) from exc


def _canonical_literal(pattern: LiteralPattern, subject_type: Type) -> Constructor:
    literal = pattern.literal
    if isinstance(subject_type, BoolType) and isinstance(literal, BoolLit):
        return BoolConstructor(literal.value)
    if isinstance(subject_type, (IntType, DecimalType)) and isinstance(
        literal, (IntLit, DecimalLit)
    ):
        return LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal(literal.value))
    if isinstance(subject_type, TextType) and isinstance(literal, StringLit):
        return LiteralConstructor(LiteralKind.TEXT, literal.value)
    if isinstance(subject_type, JsonType) and isinstance(literal, NullLit):
        return LiteralConstructor(LiteralKind.NULL, None)
    raise MatchCompileInvariantError(
        "checked literal pattern is incompatible with its occurrence type: "
        f"{type(literal).__name__} against {subject_type!r}"
    )


def _normalize_pattern(
    pattern: Pattern,
    subject_type: Type,
    checked: CheckedPatternOwner,
) -> PatternCell:
    provenance = SourcePatternProvenance(pattern.node_id, pattern.span)
    match pattern:
        case WildcardPattern():
            return WildcardCell(binder=None, provenance=provenance)
        case VarPattern(node_id=node_id, name=name):
            if node_id not in checked.resolved.bare_variant_patterns:
                return WildcardCell(
                    binder=BinderProvenance(node_id=node_id, name=name, span=pattern.span),
                    provenance=provenance,
                )
            if not isinstance(subject_type, EnumType):
                raise MatchCompileInvariantError(
                    "resolver-classified bare variant has a non-enum checked type"
                )
            constructor_ref = checked.resolved.bare_variant_refs.get(node_id)
            if constructor_ref is None or constructor_ref.variant != name:
                raise MatchCompileInvariantError(
                    "missing or inconsistent resolved constructor ref for bare variant"
                )
            constructor = _enum_constructor(
                EnumType(
                    constructor_ref.owner_name,
                    subject_type.type_args,
                    constructor_ref.owner_module_id,
                ),
                name,
                checked.type_env.type_table,
            )
            if constructor.enum_type != subject_type:
                raise MatchCompileInvariantError(
                    "resolved bare variant owner does not match its checked scrutinee type"
                )
            if constructor.arity != 0:
                raise MatchCompileInvariantError(
                    f"resolver-classified bare variant {name!r} is not nullary"
                )
            return ConstructorCell(constructor, (), provenance)
        case LiteralPattern():
            return ConstructorCell(
                constructor=_canonical_literal(pattern, subject_type),
                arguments=(),
                provenance=provenance,
            )
        case ConstructorPattern(name=variant):
            if not isinstance(subject_type, EnumType):
                raise MatchCompileInvariantError(
                    "checked constructor pattern has a non-enum occurrence type"
                )
            constructor = _enum_constructor(
                subject_type, variant, checked.type_env.type_table
            )
            supplied_pairs = checked.argument_bindings.constructor_patterns.get(pattern.node_id)
            if supplied_pairs is None:
                raise MatchCompileInvariantError(
                    f"missing checked argument bindings for pattern node {pattern.node_id}"
                )
            supplied = dict(supplied_pairs)
            if len(supplied) != len(supplied_pairs):
                raise MatchCompileInvariantError(
                    f"duplicate checked field binding for pattern node {pattern.node_id}"
                )
            declared_names = {field.name for field in constructor.fields}
            unknown = supplied.keys() - declared_names
            if unknown:
                raise MatchCompileInvariantError(
                    f"checked pattern node {pattern.node_id} binds unknown fields "
                    f"{sorted(unknown)!r}"
                )
            arguments: list[PatternCell] = []
            for field in constructor.fields:
                child = supplied.get(field.name)
                if child is None:
                    arguments.append(
                        WildcardCell(
                            binder=None,
                            provenance=OmittedFieldProvenance(
                                constructor_pattern_id=pattern.node_id,
                                field_name=field.name,
                                span=pattern.span,
                            ),
                        )
                    )
                else:
                    arguments.append(_normalize_pattern(child, field.type, checked))
            return ConstructorCell(constructor, tuple(arguments), provenance)
        case _ as unreachable:
            try:
                assert_never(unreachable)
            except AssertionError as exc:
                raise MatchCompileInvariantError(
                    f"unsupported source pattern {type(unreachable).__name__}"
                ) from exc


def normalize_pattern(
    pattern: Pattern,
    subject_type: Type,
    checked: CheckedPatternOwner,
) -> PatternCell:
    """Normalize one checked pattern against its checked occurrence type."""
    return _normalize_pattern(pattern, subject_type, checked)


def normalize_case(case: Case, checked: CheckedPatternOwner) -> NormalizedCase:
    """Normalize one checked source case into a source-priority one-column matrix."""
    try:
        subject_type = checked.node_types[case.subject.node_id]
    except KeyError as exc:
        raise MatchCompileInvariantError(
            f"missing checked subject type for case node {case.node_id}"
        ) from exc
    root = Occurrence(
        id=OccurrenceId(0),
        creation_order=0,
        type=subject_type,
        provenance=RootOccurrenceProvenance(
            case_node_id=case.node_id,
            subject_node_id=case.subject.node_id,
            span=case.subject.span,
        ),
    )
    rows: list[MatrixRow] = []
    for index, branch in enumerate(case.branches):
        cell = _normalize_pattern(branch.pattern, subject_type, checked)
        if pattern_cell_inhabits_type(cell, subject_type):
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
            body_node_id=branch.body.node_id,
            branch_span=branch.span,
            pattern_span=branch.pattern.span,
        )
        for index, branch in enumerate(case.branches)
    )
    try:
        case_scope = checked.resolved.case_scopes[case.node_id]
    except KeyError as exc:
        raise MatchCompileInvariantError(
            f"missing resolver scope provenance for case node {case.node_id}"
        ) from exc
    module_id = checked.module_id if isinstance(checked, CheckedModule) else ENTRY_ID
    case_context = MatchCaseContext(
        module_id=module_id,
        enum_owner_forms=checked.type_env.enum_owner_forms(),
        bare_enum_constructors=_bare_enum_constructors(case_scope, checked),
        owner_program=checked.resolved.program,
    )
    return NormalizedCase(
        case_node_id=case.node_id,
        span=case.span,
        root=root,
        occurrences=(root,),
        rows=tuple(rows),
        actions=actions,
        type_table=checked.type_env.type_table,
        case_context=case_context,
    )


__all__ = [
    "CheckedPatternOwner",
    "MatchCompileInvariantError",
    "constructor_inhabits_type",
    "normalize_case",
    "normalize_pattern",
    "pattern_cell_inhabits_type",
    "signature_for_type",
]
