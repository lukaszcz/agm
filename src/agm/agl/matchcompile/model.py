"""Immutable compiler-private data for AgL pattern-matrix compilation.

The model deliberately contains no execution IR.  It records checked source
patterns as canonical cells and provides the decision-node identities consumed
by later match-compilation and lowering stages.
"""

from __future__ import annotations

import decimal
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import EnumOwnerForm, EnumType, Type
from agm.agl.syntax.nodes import Program
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
    _indices: dict[Constructor, int] | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def index_of(self, constructor: Constructor) -> int | None:
        """Return the declaration index of *constructor*, or ``None`` when absent.

        The index is built once per signature so declaration-order lookups and
        completeness tests never rescan the constructor tuple.
        """
        indices = self._indices
        if indices is None:
            indices = {}
            for index, candidate in enumerate(self.constructors):
                indices.setdefault(candidate, index)
            object.__setattr__(self, "_indices", indices)
        return indices.get(constructor)


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
class PathDecomposition:
    """One constructor decomposition selected on the current compilation path."""

    parent: Occurrence
    constructor: Constructor
    children: tuple[Occurrence, ...]


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
        if self_validation_enabled():
            _validate_constructor_cell(self)


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


EnumConstructorSpelling: TypeAlias = EnumOwnerForm


@dataclass(frozen=True, slots=True)
class MatrixRow:
    """One source-priority row in a canonical pattern matrix."""

    cells: tuple[PatternCell, ...]
    action_id: int
    source_index: int
    source_pattern_id: int
    binder_assignments: tuple[BinderAssignment, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchCaseContext:
    """Per-case frontend context used only for diagnostics and allocation identity."""

    module_id: ModuleId
    enum_owner_forms: tuple[EnumOwnerForm, ...] = ()
    # Variants a same-named module route makes ambiguous under one owner
    # form's short ``(owner_name,)`` qualifier; see
    # ``TypeEnvironment.blocked_enum_variants``. Plain data, not compared: the
    # forms it corresponds to are already excluded from case-context equality.
    blocked_enum_variants: Mapping[tuple[str, ...], frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )
    bare_enum_constructors: frozenset[tuple[ModuleId, str, str]] = frozenset()
    owner_program: Program | None = field(default=None, repr=False, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class NormalizedCase:
    """The normalized one-column matrix and source identities for a source case."""

    case_node_id: int
    span: SourceSpan
    root: Occurrence
    occurrences: tuple[Occurrence, ...]
    rows: tuple[MatrixRow, ...]
    actions: tuple[SourceAction, ...]
    type_table: TypeTable = field(repr=False, compare=False, hash=False)
    case_context: MatchCaseContext = field(
        default_factory=lambda: MatchCaseContext(ENTRY_ID),
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self_validation_enabled():
            _validate_normalized_case(self)


@dataclass(frozen=True, slots=True)
class DecisionFail:
    """A path on which no source row matches."""

    @property
    def free_occurrences(self) -> tuple[OccurrenceId, ...]:
        """Occurrences which must be available when this decision is entered."""
        return ()


@dataclass(frozen=True, slots=True)
class DecisionLeaf:
    """A selected source action and its dominated binder assignments."""

    action_id: int
    binder_assignments: tuple[BinderAssignment, ...]

    @property
    def free_occurrences(self) -> tuple[OccurrenceId, ...]:
        """Occurrences read to initialize source binders before the action."""
        occurrences: list[OccurrenceId] = []
        for assignment in self.binder_assignments:
            if assignment.occurrence not in occurrences:
                occurrences.append(assignment.occurrence)
        return tuple(occurrences)


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
    free_occurrences: tuple[OccurrenceId, ...] = ()


Decision: TypeAlias = DecisionFail | DecisionLeaf | DecisionSwitch


# ---------------------------------------------------------------------------
# Optional self-validation
#
# Invariant self-checks that re-verify this module's own construction.  They
# never change the compiler's result and run only when optional
# match-compilation validation is enabled (see ``agm.agl.self_validation``); the
# test harness turns them on so every value built anywhere in the suite is
# validated.
# ---------------------------------------------------------------------------


def _validate_constructor_cell(cell: ConstructorCell) -> None:
    if len(cell.arguments) != cell.constructor.arity:
        raise ValueError(
            "constructor cell argument count does not match constructor arity: "
            f"{len(cell.arguments)} != {cell.constructor.arity}"
        )


def _validate_normalized_case(case: NormalizedCase) -> None:
    if case.occurrences != (case.root,):
        raise ValueError("a freshly normalized case must contain only its root occurrence")
    if any(len(row.cells) != len(case.occurrences) for row in case.rows):
        raise ValueError("normalized matrix row width does not match occurrence width")
    if tuple(action.source_index for action in case.actions) != tuple(range(len(case.actions))):
        raise ValueError("source actions must retain contiguous source priority")
    row_indices = tuple(row.source_index for row in case.rows)
    if row_indices != tuple(sorted(set(row_indices))) or any(
        not 0 <= source_index < len(case.actions) for source_index in row_indices
    ):
        raise ValueError(
            "normalized matrix rows must retain an ordered unique subsequence of source actions"
        )
    if any(row.action_id != case.actions[row.source_index].action_id for row in case.rows):
        raise ValueError("normalized rows and source actions must agree for each retained row")


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
    "EnumConstructorSpelling",
    "FieldOccurrenceProvenance",
    "LiteralConstructor",
    "LiteralKind",
    "LiteralValue",
    "MatchCaseContext",
    "MatrixRow",
    "NormalizedCase",
    "Occurrence",
    "OccurrenceId",
    "OccurrenceProvenance",
    "OmittedFieldProvenance",
    "OpenSignature",
    "PathDecomposition",
    "PatternCell",
    "PatternProvenance",
    "RootOccurrenceProvenance",
    "Signature",
    "SourceAction",
    "SourcePatternProvenance",
    "WildcardCell",
]
