"""Immutable compiler-private data for AgL pattern-matrix compilation.

The model deliberately contains no execution IR.  It records checked source
patterns as canonical cells and provides the decision-node identities consumed
by later match-compilation and lowering stages.
"""

from __future__ import annotations

import decimal
import enum
from dataclasses import dataclass
from typing import TypeAlias

from agm.agl.semantics.types import EnumType, Type
from agm.agl.syntax.spans import SourceSpan


@dataclass(frozen=True, slots=True, order=True)
class OccurrenceId:
    """Stable, case-local identity of a value occurrence."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("occurrence ids must be non-negative")


@dataclass(frozen=True, slots=True)
class ConstructorField:
    """One declaration-order child exposed by an enum constructor."""

    name: str
    type: Type


@dataclass(frozen=True, slots=True)
class EnumConstructor:
    """A typed enum-variant constructor head."""

    enum_type: EnumType
    variant: str
    fields: tuple[ConstructorField, ...]

    @property
    def arity(self) -> int:
        return len(self.fields)


@dataclass(frozen=True, slots=True)
class BoolConstructor:
    """One constructor in the closed boolean signature."""

    value: bool

    @property
    def arity(self) -> int:
        return 0


class LiteralKind(enum.Enum):
    """Equality domains for scalar literal constructor keys."""

    NUMERIC = "numeric"
    TEXT = "text"
    NULL = "null"


LiteralValue: TypeAlias = decimal.Decimal | str | None


@dataclass(frozen=True, slots=True)
class LiteralConstructor:
    """A canonical scalar literal constructor head.

    Numeric values are always represented as :class:`decimal.Decimal`, making
    integer and decimal literals that compare equal at runtime the same key.
    """

    kind: LiteralKind
    value: LiteralValue

    def __post_init__(self) -> None:
        valid = (
            self.kind is LiteralKind.NUMERIC
            and isinstance(self.value, decimal.Decimal)
            or self.kind is LiteralKind.TEXT
            and isinstance(self.value, str)
            or self.kind is LiteralKind.NULL
            and self.value is None
        )
        if not valid:
            raise ValueError(f"invalid value {self.value!r} for literal kind {self.kind.value}")

    @property
    def arity(self) -> int:
        return 0


Constructor: TypeAlias = EnumConstructor | BoolConstructor | LiteralConstructor


@dataclass(frozen=True, slots=True)
class ClosedSignature:
    """A finite, declaration-ordered set of all constructors for a type."""

    constructors: tuple[Constructor, ...]


@dataclass(frozen=True, slots=True)
class OpenSignature:
    """A type domain whose values are not finitely enumerated by constructors."""


Signature: TypeAlias = ClosedSignature | OpenSignature


@dataclass(frozen=True, slots=True)
class SourcePatternProvenance:
    """The source pattern node which produced a canonical cell."""

    node_id: int
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class OmittedFieldProvenance:
    """A wildcard synthesized for an omitted constructor field."""

    constructor_pattern_id: int
    field_name: str
    span: SourceSpan


PatternProvenance: TypeAlias = SourcePatternProvenance | OmittedFieldProvenance


@dataclass(frozen=True, slots=True)
class RootOccurrenceProvenance:
    """Provenance for the root scrutinee occurrence of one source case."""

    case_node_id: int
    subject_node_id: int
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class FieldOccurrenceProvenance:
    """Provenance for a declaration-order child introduced by specialization."""

    parent: OccurrenceId
    constructor: Constructor
    field_name: str
    field_index: int
    source: PatternProvenance


OccurrenceProvenance: TypeAlias = RootOccurrenceProvenance | FieldOccurrenceProvenance


@dataclass(frozen=True, slots=True)
class Occurrence:
    """A typed value available to the match compiler."""

    id: OccurrenceId
    creation_order: int
    type: Type
    provenance: OccurrenceProvenance

    def __post_init__(self) -> None:
        if self.creation_order < 0:
            raise ValueError("occurrence creation order must be non-negative")


@dataclass(frozen=True, slots=True)
class BinderProvenance:
    """Identity and source provenance of a real variable pattern."""

    node_id: int
    name: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class WildcardCell:
    """An irrefutable matrix cell, optionally annotated with a source binder."""

    binder: BinderProvenance | None
    provenance: PatternProvenance


@dataclass(frozen=True, slots=True)
class ConstructorCell:
    """A refutable head and its declaration-order child pattern cells."""

    constructor: Constructor
    arguments: tuple[PatternCell, ...]
    provenance: SourcePatternProvenance

    def __post_init__(self) -> None:
        if len(self.arguments) != self.constructor.arity:
            raise ValueError(
                "constructor cell argument count does not match constructor arity: "
                f"{len(self.arguments)} != {self.constructor.arity}"
            )


PatternCell: TypeAlias = WildcardCell | ConstructorCell


@dataclass(frozen=True, slots=True)
class BinderAssignment:
    """A leaf-time assignment from an available occurrence to a source binder."""

    occurrence: OccurrenceId
    binder: BinderProvenance


@dataclass(frozen=True, slots=True)
class SourceAction:
    """Stable identity and location of one source case arm body."""

    action_id: int
    source_index: int
    body_node_id: int
    branch_span: SourceSpan
    pattern_span: SourceSpan


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """One source-priority row in a canonical pattern matrix."""

    cells: tuple[PatternCell, ...]
    action_id: int
    source_index: int
    source_pattern_id: int
    binder_assignments: tuple[BinderAssignment, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedCase:
    """The normalized one-column matrix and source identities for a source case."""

    case_node_id: int
    span: SourceSpan
    root: Occurrence
    occurrences: tuple[Occurrence, ...]
    rows: tuple[MatrixRow, ...]
    actions: tuple[SourceAction, ...]

    def __post_init__(self) -> None:
        if self.occurrences != (self.root,):
            raise ValueError("a freshly normalized case must contain only its root occurrence")
        if any(len(row.cells) != len(self.occurrences) for row in self.rows):
            raise ValueError("normalized matrix row width does not match occurrence width")
        if tuple(row.source_index for row in self.rows) != tuple(range(len(self.rows))):
            raise ValueError("normalized matrix rows must retain contiguous source priority")
        if tuple(action.source_index for action in self.actions) != tuple(
            range(len(self.actions))
        ):
            raise ValueError("source actions must retain contiguous source priority")
        if tuple(row.action_id for row in self.rows) != tuple(
            action.action_id for action in self.actions
        ):
            raise ValueError("normalized rows and source actions must agree")


@dataclass(frozen=True, slots=True)
class DecisionFail:
    """A path on which no source row matches."""


@dataclass(frozen=True, slots=True)
class DecisionLeaf:
    """A selected source action and its dominated binder assignments."""

    action_id: int
    binder_assignments: tuple[BinderAssignment, ...]


@dataclass(frozen=True, slots=True)
class DecisionBranch:
    """One constructor-keyed edge from a decision switch."""

    constructor: Constructor
    decision: Decision


@dataclass(frozen=True, slots=True)
class DecisionSwitch:
    """A one-occurrence decision with deterministic keyed children and default."""

    occurrence: Occurrence
    keyed_children: tuple[DecisionBranch, ...]
    default: Decision | None


Decision: TypeAlias = DecisionFail | DecisionLeaf | DecisionSwitch


__all__ = [
    "BinderAssignment",
    "BinderProvenance",
    "BoolConstructor",
    "ClosedSignature",
    "Constructor",
    "ConstructorCell",
    "ConstructorField",
    "Decision",
    "DecisionBranch",
    "DecisionFail",
    "DecisionLeaf",
    "DecisionSwitch",
    "EnumConstructor",
    "FieldOccurrenceProvenance",
    "LiteralConstructor",
    "LiteralKind",
    "LiteralValue",
    "MatrixRow",
    "NormalizedCase",
    "Occurrence",
    "OccurrenceId",
    "OccurrenceProvenance",
    "OmittedFieldProvenance",
    "OpenSignature",
    "PatternCell",
    "PatternProvenance",
    "RootOccurrenceProvenance",
    "Signature",
    "SourceAction",
    "SourcePatternProvenance",
    "WildcardCell",
]
