"""Structured diagnostics reconstructed from compiled pattern-decision DAGs."""

from __future__ import annotations

import decimal
import json
from dataclasses import dataclass
from typing import TypeAlias

from agm.agl.semantics.types import EnumType, Type
from agm.agl.syntax.spans import SourceSpan

from .model import LiteralConstructor, LiteralKind


@dataclass(frozen=True, slots=True)
class WildcardWitness:
    """An unconstrained child of a structural missing pattern."""


@dataclass(frozen=True, slots=True)
class BoolWitness:
    """A concrete missing boolean value."""

    value: bool


@dataclass(frozen=True, slots=True)
class LiteralWitness:
    """A concrete scalar literal required by a failure path."""

    kind: LiteralKind
    value: decimal.Decimal | str | None


@dataclass(frozen=True, slots=True)
class WitnessField:
    """One declaration-order field of an enum witness."""

    name: str
    witness: MatchWitness


@dataclass(frozen=True, slots=True)
class EnumWitnessQualification:
    """Source-level enum owner spelling used to render one witness.

    This diagnostic value deliberately omits the type-checker's owner-form
    resolution metadata.  Consumers need only the selected owner name and its
    optional module qualifier.
    """

    owner_name: str
    module_qualifier: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class EnumWitness:
    """A concrete enum constructor with structural child witnesses."""

    enum_type: EnumType
    variant: str
    fields: tuple[WitnessField, ...]
    qualification: EnumWitnessQualification | None = None


@dataclass(frozen=True, slots=True)
class OpenComplementWitness:
    """The remainder of an open domain after excluding observed literals."""

    subject_type: Type
    excluded: tuple[LiteralConstructor, ...]


MatchWitness: TypeAlias = (
    WildcardWitness | BoolWitness | LiteralWitness | EnumWitness | OpenComplementWitness
)


@dataclass(frozen=True, slots=True)
class NonExhaustiveIssue:
    """One source case has a reachable failure path."""

    case_node_id: int
    span: SourceSpan
    witness: MatchWitness


@dataclass(frozen=True, slots=True)
class RedundantArmIssue:
    """One source arm action is unreachable in the compiled decision DAG."""

    case_node_id: int
    action_id: int
    span: SourceSpan


MatchIssue: TypeAlias = NonExhaustiveIssue | RedundantArmIssue


def _render_literal(kind: LiteralKind, value: decimal.Decimal | str | None) -> str:
    if kind is LiteralKind.TEXT:
        assert isinstance(value, str)
        return json.dumps(value, ensure_ascii=False).replace("${", "\\${")
    if kind is LiteralKind.NULL:
        return "null"
    assert isinstance(value, decimal.Decimal)
    return format(value, "f")


def render_witness(witness: MatchWitness) -> str:
    """Render structured witness data for a later user-facing diagnostic adapter."""
    if isinstance(witness, WildcardWitness):
        return "_"
    if isinstance(witness, BoolWitness):
        return "true" if witness.value else "false"
    if isinstance(witness, LiteralWitness):
        return _render_literal(witness.kind, witness.value)
    if isinstance(witness, EnumWitness):
        constructor_name = witness.variant
        if witness.qualification is not None:
            owner_name = witness.qualification.owner_name
            assert owner_name is not None
            qualifier = witness.qualification.module_qualifier
            if qualifier is None:
                module_prefix = ""
            elif qualifier:
                module_prefix = f"{'.'.join(qualifier)}::"
            else:
                module_prefix = "::"
            constructor_name = f"{module_prefix}{owner_name}::{witness.variant}"
        if not witness.fields:
            return constructor_name
        fields = ", ".join(
            f"{field.name} = {render_witness(field.witness)}" for field in witness.fields
        )
        return f"{constructor_name}({fields})"
    excluded = ", ".join(
        _render_literal(constructor.kind, constructor.value) for constructor in witness.excluded
    )
    domain = repr(witness.subject_type)
    if not excluded:
        return f"a {domain} value"
    return f"a {domain} value other than {excluded}"


def issue_sort_key(issue: MatchIssue) -> tuple[str, int, int, int, int, int, int]:
    """Return the deterministic cross-source ordering key used by the stage adapter."""
    kind_order = 0 if isinstance(issue, NonExhaustiveIssue) else 1
    action_id = -1 if isinstance(issue, NonExhaustiveIssue) else issue.action_id
    span = issue.span
    return (
        span.source.label,
        span.start_offset,
        span.end_offset,
        span.start_line,
        span.start_col,
        kind_order,
        action_id,
    )


__all__ = [
    "BoolWitness",
    "EnumWitness",
    "EnumWitnessQualification",
    "LiteralWitness",
    "MatchIssue",
    "MatchWitness",
    "NonExhaustiveIssue",
    "OpenComplementWitness",
    "RedundantArmIssue",
    "WildcardWitness",
    "WitnessField",
    "issue_sort_key",
    "render_witness",
]
