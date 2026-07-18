"""Small source-pattern reference matcher for match-compiler tests.

This helper intentionally interprets checked source patterns directly.  It is
kept independent of the production pattern-matrix implementation so generated
compiler tests can compare selected source actions against a simple oracle.
"""

from __future__ import annotations

import decimal
from typing import assert_never

from agm.agl.eval.arith import value_eq
from agm.agl.matchcompile.matrix import PatternMatrix
from agm.agl.matchcompile.model import (
    BoolConstructor,
    EnumConstructor,
    LiteralConstructor,
    LiteralKind,
    PatternCell,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import CheckedPatternOwner
from agm.agl.semantics.types import EnumType, Type
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    EnumValue,
    IntValue,
    JsonValue,
    TextValue,
    Value,
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


def _literal_value(pattern: LiteralPattern) -> Value:
    literal = pattern.literal
    match literal:
        case IntLit(value=value):
            return IntValue(value)
        case DecimalLit(value=value):
            return DecimalValue(value)
        case BoolLit(value=value):
            return BoolValue(value)
        case StringLit(value=value):
            return TextValue(value)
        case NullLit():
            return JsonValue(None)
        case _ as unreachable:
            assert_never(unreachable)


def _matches(
    pattern: Pattern,
    subject_type: Type,
    value: Value,
    checked: CheckedPatternOwner,
) -> bool:
    match pattern:
        case WildcardPattern():
            return True
        case AsPattern(pattern=inner):
            return _matches(inner, subject_type, value, checked)
        case VarPattern(node_id=node_id, name=variant):
            if node_id in checked.argument_bindings.pattern_binders:
                return True
            return (
                node_id in checked.argument_bindings.pattern_constructors
                and isinstance(subject_type, EnumType)
                and isinstance(value, EnumValue)
                and value.nominal.module_id == subject_type.module_id
                and value.nominal.declared_name == subject_type.name
                and value.variant == variant
            )
        case LiteralPattern():
            return value_eq(value, _literal_value(pattern))
        case ConstructorPattern(node_id=node_id, name=variant):
            if not isinstance(subject_type, EnumType) or not isinstance(value, EnumValue):
                return False
            if (
                value.nominal.module_id != subject_type.module_id
                or value.nominal.declared_name != subject_type.name
                or value.variant != variant
            ):
                return False
            fields = checked.type_env.type_table.enum_variants(subject_type)[variant]
            return all(
                _matches(child, fields[field_name], value.fields[field_name], checked)
                for field_name, child in checked.argument_bindings.constructor_patterns[node_id]
            )
        case _ as unreachable:
            assert_never(unreachable)


def reference_action(case: Case, checked: CheckedPatternOwner, subject: Value) -> int | None:
    """Return the first matching ``CaseBranch.node_id``, or ``None``."""
    subject_type = checked.node_types[case.subject.node_id]
    for branch in case.branches:
        if _matches(branch.pattern, subject_type, subject, checked):
            return branch.node_id
    return None


def _constructor_literal_value(constructor: LiteralConstructor) -> Value:
    if constructor.kind is LiteralKind.NUMERIC:
        assert isinstance(constructor.value, decimal.Decimal)
        return DecimalValue(constructor.value)
    if constructor.kind is LiteralKind.TEXT:
        assert isinstance(constructor.value, str)
        return TextValue(constructor.value)
    return JsonValue(None)


def canonical_cell_matches(cell: PatternCell, value: Value) -> bool:
    """Match one canonical matrix cell with AgL runtime equality semantics."""
    if isinstance(cell, WildcardCell):
        return True

    constructor = cell.constructor
    if isinstance(constructor, BoolConstructor):
        return isinstance(value, BoolValue) and value.value is constructor.value
    if isinstance(constructor, LiteralConstructor):
        return value_eq(value, _constructor_literal_value(constructor))
    if not isinstance(constructor, EnumConstructor) or not isinstance(value, EnumValue):
        return False
    if (
        value.nominal.module_id != constructor.enum_type.module_id
        or value.nominal.declared_name != constructor.enum_type.name
        or value.variant != constructor.variant
    ):
        return False
    return all(
        canonical_cell_matches(argument, value.fields[field.name])
        for field, argument in zip(constructor.fields, cell.arguments, strict=True)
    )


def matrix_action(matrix: PatternMatrix, values: tuple[Value, ...]) -> int | None:
    """Return the first action selected by a canonical pattern matrix."""
    assert len(values) == len(matrix.occurrences)
    for row in matrix.rows:
        if all(
            canonical_cell_matches(cell, value)
            for cell, value in zip(row.cells, values, strict=True)
        ):
            return row.action_id
    return None


__all__ = ["canonical_cell_matches", "matrix_action", "reference_action"]
