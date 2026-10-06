"""Structured diagnostics reconstructed from compiled pattern-decision DAGs."""

from __future__ import annotations

import decimal
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

from agm.agl.diagnostics import AglError
from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.type_table import DeclKey
from agm.agl.semantics.types import Type
from agm.agl.syntax.spans import SourceSpan
from agm.agl.value_syntax.lexical import quote_text

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


ConstructorSpeller: TypeAlias = Callable[[DeclKey, int], str]
"""Spells a constructor, by its declaration, as written in the ``case`` with the given node id."""

ConstructorSpellers: TypeAlias = Callable[[ModuleId], ConstructorSpeller]
"""A module's :data:`ConstructorSpeller`."""


@dataclass(frozen=True, slots=True)
class EnumWitness:
    """A concrete enum constructor with structural child witnesses."""

    constructor: DeclKey
    case_node_id: int
    fields: tuple[WitnessField, ...]


@dataclass(frozen=True, slots=True)
class RecordWitness:
    """A concrete record constructor with structural child witnesses."""

    constructor: DeclKey
    case_node_id: int
    fields: tuple[WitnessField, ...]


@dataclass(frozen=True, slots=True)
class OpenComplementWitness:
    """The remainder of an open domain after excluding observed literals."""

    subject_type: Type
    excluded: tuple[LiteralConstructor, ...]


MatchWitness: TypeAlias = (
    WildcardWitness
    | BoolWitness
    | LiteralWitness
    | EnumWitness
    | RecordWitness
    | OpenComplementWitness
)


@dataclass(frozen=True, slots=True)
class NonExhaustiveIssue:
    """One source case has a reachable failure path."""

    site_node_id: int
    span: SourceSpan
    witness: MatchWitness
    module_id: ModuleId


@dataclass(frozen=True, slots=True)
class RedundantArmIssue:
    """One source arm action is unreachable in the compiled decision DAG."""

    site_node_id: int
    action_id: int
    span: SourceSpan


MatchIssue: TypeAlias = NonExhaustiveIssue | RedundantArmIssue


class NonExhaustiveMatchError(AglError):
    """Static error for a :class:`NonExhaustiveIssue`: one case has a reachable failure path."""


class RedundantArmError(AglError):
    """Static error for a :class:`RedundantArmIssue`: one arm is unreachable."""


def _render_literal(kind: LiteralKind, value: decimal.Decimal | str | None) -> str:
    if kind is LiteralKind.TEXT:
        return quote_text(cast(str, value))
    if kind is LiteralKind.NULL:
        return "null"
    return format(cast(decimal.Decimal, value), "f")


def render_witness(witness: MatchWitness, speller: ConstructorSpeller) -> str:
    """Render structured witness data, spelling constructors with *speller*."""
    if isinstance(witness, WildcardWitness):
        return "_"
    if isinstance(witness, BoolWitness):
        return "true" if witness.value else "false"
    if isinstance(witness, LiteralWitness):
        return _render_literal(witness.kind, witness.value)
    if isinstance(witness, (EnumWitness, RecordWitness)):
        constructor_name = speller(witness.constructor, witness.case_node_id)
        if not witness.fields:
            return constructor_name
        fields = ", ".join(
            f"{field.name} = {render_witness(field.witness, speller)}" for field in witness.fields
        )
        return f"{constructor_name}({fields})"
    excluded = ", ".join(
        _render_literal(constructor.kind, constructor.value) for constructor in witness.excluded
    )
    domain = repr(witness.subject_type)
    if not excluded:
        return f"a {domain} value"
    return f"a {domain} value other than {excluded}"


def match_issue_error(issue: MatchIssue, spellers: ConstructorSpellers) -> AglError:
    """Build the one real static error a compiled match issue is reported as.

    *spellers* spell the constructors of a missing pattern in the module of its ``case``.

    The sole synthesis of a match issue's user-facing message and span, so
    every consumer (a raised failure, or a rendered ``Diagnostic`` via
    :func:`~agm.agl.matchcompile.stage.diagnostic_from_match_issue`) derives
    from this.
    """
    if isinstance(issue, NonExhaustiveIssue):
        missing = render_witness(issue.witness, spellers(issue.module_id))
        message = f"Non-exhaustive case; missing pattern: {missing}."
        return NonExhaustiveMatchError(message, span=issue.span)
    message = "Redundant case arm; this pattern can never be selected."
    return RedundantArmError(message, span=issue.span)


def issue_sort_key(issue: MatchIssue) -> tuple[str, int, int, int, int, int, int]:
    """Return the deterministic cross-source ordering key used by the stage adapter."""
    if isinstance(issue, NonExhaustiveIssue):
        kind_order = 0
        action_id = -1
    else:
        kind_order = 1
        action_id = issue.action_id
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
    "ConstructorSpeller",
    "ConstructorSpellers",
    "EnumWitness",
    "LiteralWitness",
    "MatchIssue",
    "MatchWitness",
    "NonExhaustiveIssue",
    "NonExhaustiveMatchError",
    "OpenComplementWitness",
    "RedundantArmIssue",
    "RedundantArmError",
    "RecordWitness",
    "WildcardWitness",
    "WitnessField",
    "issue_sort_key",
    "match_issue_error",
    "render_witness",
]
