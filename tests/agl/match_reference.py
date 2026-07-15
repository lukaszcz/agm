"""Small source-pattern reference matcher for match-compiler tests.

This helper intentionally interprets checked source patterns directly.  It is
kept independent of the production pattern-matrix implementation so generated
compiler tests can compare selected source actions against a simple oracle.
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.eval.arith import value_eq
from agm.agl.matchcompile import CheckedPatternOwner
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
        case VarPattern(node_id=node_id, name=variant):
            if node_id not in checked.resolved.bare_variant_patterns:
                return True
            return (
                isinstance(subject_type, EnumType)
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


__all__ = ["reference_action"]
