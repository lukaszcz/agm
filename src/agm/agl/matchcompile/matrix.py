"""Immutable pattern-matrix decomposition and qba column selection."""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TypeAlias

from agm.agl.modules.ids import ENTRY_ID
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    EnumType,
    Type,
)

from .model import (
    BinderAssignment,
    BoolConstructor,
    Constructor,
    ConstructorCell,
    ConstructorField,
    EnumConstructor,
    FieldOccurrenceProvenance,
    LiteralKind,
    MatchCaseContext,
    MatrixRow,
    NormalizedCase,
    Occurrence,
    OccurrenceId,
    PathDecomposition,
    PatternCell,
    PatternProvenance,
    WildcardCell,
)
from .normalize import (
    MatchCompileInvariantError,
    constructor_inhabits_type,
)
from .optional_validation import run_optional_validation


@dataclass(frozen=True, slots=True)
class _EnumConstructorKey:
    enum_type: EnumType
    variant: str


@dataclass(frozen=True, slots=True)
class _BoolConstructorKey:
    value: bool


@dataclass(frozen=True, slots=True)
class _LiteralConstructorKey:
    kind: LiteralKind
    value: decimal.Decimal | str | None


_ConstructorKey: TypeAlias = _EnumConstructorKey | _BoolConstructorKey | _LiteralConstructorKey
_ConstructorSortKey: TypeAlias = tuple[
    int,
    tuple[str, ...],
    str,
    tuple[str, ...],
    str,
]


def _constructor_key(constructor: Constructor) -> _ConstructorKey:
    if isinstance(constructor, EnumConstructor):
        return _EnumConstructorKey(constructor.enum_type, constructor.variant)
    if isinstance(constructor, BoolConstructor):
        return _BoolConstructorKey(constructor.value)
    return _LiteralConstructorKey(constructor.kind, constructor.value)


def _constructor_sort_key(constructor: Constructor) -> _ConstructorSortKey:
    """Return a total, stable ordering key for a constructor's semantic identity."""
    if isinstance(constructor, EnumConstructor):
        return (
            0,
            constructor.enum_type.module_id.segments,
            constructor.enum_type.name,
            tuple(repr(argument) for argument in constructor.enum_type.type_args),
            constructor.variant,
        )
    if isinstance(constructor, BoolConstructor):
        return (1, (), "", (), "true" if constructor.value else "false")
    return (2, (), constructor.kind.value, (), repr(constructor.value))


def _canonical_constructor(
    constructor: Constructor, subject_type: Type, type_table: TypeTable
) -> Constructor:
    if not constructor_inhabits_type(constructor, subject_type):
        raise MatchCompileInvariantError(
            f"constructor is incompatible with or uninhabited by occurrence type "
            f"{subject_type!r}"
        )
    if not isinstance(constructor, EnumConstructor):
        return constructor

    assert isinstance(subject_type, EnumType)
    try:
        fields = type_table.enum_variants(subject_type).get(constructor.variant)
    except (KeyError, AssertionError) as exc:
        raise MatchCompileInvariantError(
            f"cannot resolve enum signature for checked type {subject_type!r}"
        ) from exc
    if fields is None:
        raise MatchCompileInvariantError(
            "enum constructor does not exactly match its checked signature"
        )
    canonical = EnumConstructor(
        subject_type,
        constructor.variant,
        tuple(ConstructorField(name, field_type) for name, field_type in fields.items()),
    )
    if canonical != constructor:
        raise MatchCompileInvariantError(
            "enum constructor does not exactly match its checked signature"
        )
    return canonical


def _validate_cell(cell: PatternCell, subject_type: Type, type_table: TypeTable) -> None:
    if isinstance(cell, WildcardCell):
        return
    constructor = _canonical_constructor(cell.constructor, subject_type, type_table)
    if isinstance(constructor, EnumConstructor):
        for field, argument in zip(constructor.fields, cell.arguments, strict=True):
            _validate_cell(argument, field.type, type_table)


def _cell_binder_ids(cell: PatternCell) -> tuple[int, ...]:
    if isinstance(cell, WildcardCell):
        return () if cell.binder is None else (cell.binder.node_id,)
    return tuple(
        binder_id for argument in cell.arguments for binder_id in _cell_binder_ids(argument)
    )


@dataclass(frozen=True, slots=True)
class PatternMatrix:
    """One immutable compilation state with active and path-available occurrences."""

    occurrences: tuple[Occurrence, ...]
    rows: tuple[MatrixRow, ...]
    available_occurrences: tuple[Occurrence, ...]
    type_table: TypeTable = dataclass_field(repr=False, compare=False, hash=False)
    path_decompositions: tuple[PathDecomposition, ...] = ()
    case_context: MatchCaseContext = dataclass_field(
        default_factory=lambda: MatchCaseContext(ENTRY_ID),
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        _canonicalize_matrix(self)
        run_optional_validation(lambda: _validate_matrix(self))


def _occurrence_sort_key(occurrence: Occurrence) -> tuple[int, int]:
    return occurrence.creation_order, occurrence.id.value


def _decomposition_sort_key(
    decomposition: PathDecomposition,
) -> tuple[int, int, _ConstructorSortKey]:
    return (
        decomposition.parent.creation_order,
        decomposition.parent.id.value,
        _constructor_sort_key(decomposition.constructor),
    )


def _binder_assignment_sort_key(
    assignment: BinderAssignment,
    available_by_id: dict[OccurrenceId, Occurrence],
) -> tuple[int, int, int]:
    occurrence = available_by_id.get(assignment.occurrence)
    creation_order = (
        occurrence.creation_order if occurrence is not None else assignment.occurrence.value
    )
    return creation_order, assignment.occurrence.value, assignment.binder.node_id


def _canonicalize_matrix(matrix: PatternMatrix) -> None:
    """Canonicalize order-insensitive state before a frozen matrix can escape."""
    available_occurrences = tuple(
        sorted(matrix.available_occurrences, key=_occurrence_sort_key)
    )
    available_by_id = {
        occurrence.id: occurrence for occurrence in available_occurrences
    }

    def binder_assignment_key(assignment: BinderAssignment) -> tuple[int, int, int]:
        return _binder_assignment_sort_key(assignment, available_by_id)

    rows = tuple(
        MatrixRow(
            cells=row.cells,
            action_id=row.action_id,
            source_index=row.source_index,
            source_pattern_id=row.source_pattern_id,
            binder_assignments=tuple(
                sorted(
                    row.binder_assignments,
                    key=binder_assignment_key,
                )
            ),
        )
        for row in matrix.rows
    )
    path_decompositions = tuple(
        sorted(matrix.path_decompositions, key=_decomposition_sort_key)
    )
    object.__setattr__(matrix, "available_occurrences", available_occurrences)
    object.__setattr__(matrix, "rows", rows)
    object.__setattr__(matrix, "path_decompositions", path_decompositions)


def _validate_matrix(matrix: PatternMatrix) -> None:
    available_by_id: dict[OccurrenceId, Occurrence] = {}
    for occurrence in matrix.available_occurrences:
        if occurrence.id in available_by_id:
            raise MatchCompileInvariantError("available occurrence ids must be unique")
        available_by_id[occurrence.id] = occurrence

    active_ids: set[OccurrenceId] = set()
    for occurrence in matrix.occurrences:
        if occurrence.id in active_ids:
            raise MatchCompileInvariantError("active occurrence ids must be unique")
        active_ids.add(occurrence.id)
        if available_by_id.get(occurrence.id) != occurrence:
            raise MatchCompileInvariantError(
                f"active occurrence {occurrence.id.value} is not available on this path"
            )

    _validate_path_decompositions(matrix, available_by_id, active_ids)

    source_indices = tuple(row.source_index for row in matrix.rows)
    if any(later < earlier for earlier, later in zip(source_indices, source_indices[1:])):
        raise MatchCompileInvariantError("matrix rows do not retain source order")

    for row in matrix.rows:
        if len(row.cells) != len(matrix.occurrences):
            raise MatchCompileInvariantError(
                "matrix row width does not match occurrence-vector width"
            )
        binder_ids = [binder_id for cell in row.cells for binder_id in _cell_binder_ids(cell)]
        for assignment in row.binder_assignments:
            if assignment.occurrence not in available_by_id:
                raise MatchCompileInvariantError(
                    f"binder assignment refers to unavailable occurrence "
                    f"{assignment.occurrence.value}"
                )
            binder_ids.append(assignment.binder.node_id)
        if len(set(binder_ids)) != len(binder_ids):
            raise MatchCompileInvariantError("a row binds the same source binder more than once")

        for cell, occurrence in zip(row.cells, matrix.occurrences, strict=True):
            _validate_cell(cell, occurrence.type, matrix.type_table)


def _validate_path_decompositions(
    matrix: PatternMatrix,
    available_by_id: dict[OccurrenceId, Occurrence],
    active_ids: set[OccurrenceId],
) -> None:
    child_owner: dict[OccurrenceId, int] = {}
    decomposed_parents: set[OccurrenceId] = set()

    for decomposition_index, decomposition in enumerate(matrix.path_decompositions):
        parent = available_by_id.get(decomposition.parent.id)
        if parent != decomposition.parent:
            raise MatchCompileInvariantError(
                "path decomposition parent is not exactly available on this path"
            )
        if decomposition.parent.id in decomposed_parents:
            raise MatchCompileInvariantError(
                f"path decomposition tests parent {decomposition.parent.id.value} more than once"
            )
        canonical_constructor = _canonical_constructor(
            decomposition.constructor,
            decomposition.parent.type,
            matrix.type_table,
        )

        parent_provenance = decomposition.parent.provenance
        if isinstance(parent_provenance, FieldOccurrenceProvenance):
            if decomposition.parent.id not in child_owner:
                raise MatchCompileInvariantError(
                    "path decomposition parent is not dominated by an earlier decomposition"
                )

        fields = (
            canonical_constructor.fields
            if isinstance(canonical_constructor, EnumConstructor)
            else ()
        )
        if len(decomposition.children) != len(fields):
            raise MatchCompileInvariantError(
                "path decomposition child group is incomplete for its constructor"
            )
        for field_index, (field, child) in enumerate(
            zip(fields, decomposition.children, strict=True)
        ):
            if available_by_id.get(child.id) != child:
                raise MatchCompileInvariantError(
                    "path decomposition child is not exactly available on this path"
                )
            if child.id in child_owner:
                raise MatchCompileInvariantError(
                    f"field occurrence {child.id.value} belongs to multiple path decompositions"
                )
            provenance = child.provenance
            if (
                not isinstance(provenance, FieldOccurrenceProvenance)
                or provenance.parent != decomposition.parent.id
                or provenance.constructor != decomposition.constructor
                or provenance.field_name != field.name
                or provenance.field_index != field_index
                or child.type != field.type
            ):
                raise MatchCompileInvariantError(
                    f"field occurrence {child.id.value} does not match its path decomposition "
                    "in exact declaration order"
                )
            child_owner[child.id] = decomposition_index

        decomposed_parents.add(decomposition.parent.id)

    for occurrence in matrix.available_occurrences:
        if (
            isinstance(occurrence.provenance, FieldOccurrenceProvenance)
            and occurrence.id not in child_owner
        ):
            raise MatchCompileInvariantError(
                f"field occurrence {occurrence.id.value} does not belong to a path decomposition"
            )

    tested_and_active = decomposed_parents & active_ids
    if tested_and_active:
        identifier = min(tested_and_active)
        raise MatchCompileInvariantError(
            f"decomposed parent occurrence {identifier.value} remains active"
        )


def validate_matrix(matrix: PatternMatrix) -> None:
    """Recheck all matrix operation-boundary invariants."""
    _validate_matrix(matrix)


def matrix_from_normalized(case: NormalizedCase) -> PatternMatrix:
    """Construct the initial matrix state for a normalized source case."""
    return PatternMatrix(
        case.occurrences,
        case.rows,
        case.occurrences,
        case.type_table,
        case_context=case.case_context,
    )


@dataclass(frozen=True, slots=True)
class _OccurrenceAllocation:
    parent: OccurrenceId
    constructor_key: _ConstructorKey
    children: tuple[Occurrence, ...]


@dataclass(frozen=True, slots=True)
class OccurrenceAllocator:
    """Persistent case-local allocator for stable structural child occurrences."""

    initial_occurrences: tuple[Occurrence, ...]
    allocations: tuple[_OccurrenceAllocation, ...]
    next_id: int
    next_creation_order: int
    type_table: TypeTable = dataclass_field(repr=False, compare=False, hash=False)
    case_context: MatchCaseContext = dataclass_field(
        repr=False, compare=False, hash=False
    )

    @classmethod
    def for_case(cls, case: NormalizedCase) -> OccurrenceAllocator:
        """Create the sole root allocator for one normalized source case."""
        if not isinstance(case, NormalizedCase):
            raise MatchCompileInvariantError(
                "occurrence allocator requires a normalized case root"
            )
        next_id = (
            max(
                (occurrence.id.value for occurrence in case.occurrences),
                default=-1,
            )
            + 1
        )
        next_order = (
            max(
                (occurrence.creation_order for occurrence in case.occurrences),
                default=-1,
            )
            + 1
        )
        return cls(
            case.occurrences,
            (),
            next_id,
            next_order,
            case.type_table,
            case.case_context,
        )

    @property
    def occurrences(self) -> tuple[Occurrence, ...]:
        """Return every case-local occurrence allocated so far in creation order."""
        known = _known_occurrences(self)
        return tuple(sorted(known.values(), key=_occurrence_sort_key))


def _known_occurrences(allocator: OccurrenceAllocator) -> dict[OccurrenceId, Occurrence]:
    known = {occurrence.id: occurrence for occurrence in allocator.initial_occurrences}
    for allocation in allocator.allocations:
        known.update((occurrence.id, occurrence) for occurrence in allocation.children)
    return known


def _validate_allocator(matrix: PatternMatrix, allocator: OccurrenceAllocator) -> None:
    if allocator.case_context is not matrix.case_context:
        raise MatchCompileInvariantError(
            "occurrence allocator belongs to a different case compilation"
        )
    if allocator.type_table is not matrix.type_table:
        raise MatchCompileInvariantError(
            "occurrence allocator belongs to a different compiler context"
        )
    known = _known_occurrences(allocator)
    if any(known.get(occurrence.id) != occurrence for occurrence in matrix.available_occurrences):
        raise MatchCompileInvariantError(
            "occurrence allocator does not belong to this matrix compilation"
        )
    if allocator.next_id <= max((identifier.value for identifier in known), default=-1):
        raise MatchCompileInvariantError("occurrence allocator next id is not fresh")
    if allocator.next_creation_order <= max(
        (occurrence.creation_order for occurrence in known.values()), default=-1
    ):
        raise MatchCompileInvariantError("occurrence allocator next creation order is not fresh")


def _allocate_children(
    allocator: OccurrenceAllocator,
    parent: Occurrence,
    constructor: Constructor,
    sources: tuple[PatternProvenance, ...],
) -> tuple[tuple[Occurrence, ...], OccurrenceAllocator]:
    key = _constructor_key(constructor)
    for allocation in allocator.allocations:
        if allocation.parent == parent.id and allocation.constructor_key == key:
            return allocation.children, allocator

    if not isinstance(constructor, EnumConstructor):
        children: tuple[Occurrence, ...] = ()
    else:
        children = tuple(
            Occurrence(
                id=OccurrenceId(allocator.next_id + index),
                creation_order=allocator.next_creation_order + index,
                type=field.type,
                provenance=FieldOccurrenceProvenance(
                    parent=parent.id,
                    constructor=constructor,
                    field_name=field.name,
                    field_index=index,
                    source=sources[index],
                ),
            )
            for index, field in enumerate(constructor.fields)
        )
    allocation = _OccurrenceAllocation(parent.id, key, children)
    return children, OccurrenceAllocator(
        initial_occurrences=allocator.initial_occurrences,
        allocations=(*allocator.allocations, allocation),
        next_id=allocator.next_id + constructor.arity,
        next_creation_order=allocator.next_creation_order + constructor.arity,
        type_table=allocator.type_table,
        case_context=allocator.case_context,
    )


def _check_column(matrix: PatternMatrix, column: int) -> None:
    if not 0 <= column < len(matrix.occurrences):
        raise MatchCompileInvariantError(
            f"matrix column {column} is outside width {len(matrix.occurrences)}"
        )


def _validate_operation(
    matrix: PatternMatrix,
    *,
    column: int | None = None,
    allocator: OccurrenceAllocator | None = None,
) -> None:
    """Re-check the operation-boundary invariants of one matrix operation."""
    validate_matrix(matrix)
    if column is not None:
        _check_column(matrix, column)
    if allocator is not None:
        _validate_allocator(matrix, allocator)


def _head_constructors(matrix: PatternMatrix, column: int) -> tuple[Constructor, ...]:
    """Return heads from an already validated matrix and column."""
    heads: list[Constructor] = []
    seen: set[_ConstructorKey] = set()
    for row in matrix.rows:
        cell = row.cells[column]
        if not isinstance(cell, ConstructorCell):
            continue
        key = _constructor_key(cell.constructor)
        if key not in seen:
            heads.append(cell.constructor)
            seen.add(key)
    return tuple(heads)


def head_constructors(matrix: PatternMatrix, column: int) -> tuple[Constructor, ...]:
    """Return distinct observed heads in stable first-observation order."""
    run_optional_validation(lambda: _validate_operation(matrix, column=column))
    return _head_constructors(matrix, column)


def _migrate_binder(
    row: MatrixRow, cell: WildcardCell, occurrence: Occurrence
) -> tuple[BinderAssignment, ...]:
    if cell.binder is None:
        return row.binder_assignments
    return (
        *row.binder_assignments,
        BinderAssignment(occurrence=occurrence.id, binder=cell.binder),
    )


@dataclass(frozen=True, slots=True)
class Specialization:
    """A specialized matrix together with its persistent allocator state."""

    matrix: PatternMatrix
    allocator: OccurrenceAllocator


def specialize(
    matrix: PatternMatrix,
    column: int,
    constructor: Constructor,
    allocator: OccurrenceAllocator,
) -> Specialization:
    """Specialize one selected occurrence for an observed constructor head."""
    run_optional_validation(
        lambda: _validate_operation(matrix, column=column, allocator=allocator)
    )
    observed_keys = {_constructor_key(head) for head in _head_constructors(matrix, column)}
    if _constructor_key(constructor) not in observed_keys:
        raise MatchCompileInvariantError(
            "cannot specialize a constructor head not observed in the selected column"
        )
    selected = matrix.occurrences[column]
    canonical = _canonical_constructor(constructor, selected.type, matrix.type_table)
    first_cell = next(
        cast_cell
        for row in matrix.rows
        if isinstance((cast_cell := row.cells[column]), ConstructorCell)
        and _constructor_key(cast_cell.constructor) == _constructor_key(constructor)
    )
    sources = tuple(argument.provenance for argument in first_cell.arguments)
    children, next_allocator = _allocate_children(allocator, selected, canonical, sources)

    rows: list[MatrixRow] = []
    for row in matrix.rows:
        cell = row.cells[column]
        prefix = row.cells[:column]
        suffix = row.cells[column + 1 :]
        if isinstance(cell, ConstructorCell):
            if _constructor_key(cell.constructor) != _constructor_key(constructor):
                continue
            replacement = cell.arguments
            assignments = row.binder_assignments
        else:
            replacement = tuple(
                WildcardCell(binder=None, provenance=cell.provenance)
                for _ in range(canonical.arity)
            )
            assignments = _migrate_binder(row, cell, selected)
        rows.append(
            MatrixRow(
                cells=(*prefix, *replacement, *suffix),
                action_id=row.action_id,
                source_index=row.source_index,
                source_pattern_id=row.source_pattern_id,
                binder_assignments=assignments,
            )
        )

    occurrences = (
        *matrix.occurrences[:column],
        *children,
        *matrix.occurrences[column + 1 :],
    )
    return Specialization(
        PatternMatrix(
            occurrences=occurrences,
            rows=tuple(rows),
            available_occurrences=(*matrix.available_occurrences, *children),
            path_decompositions=(
                *matrix.path_decompositions,
                PathDecomposition(selected, canonical, children),
            ),
            type_table=matrix.type_table,
            case_context=matrix.case_context,
        ),
        next_allocator,
    )


def default_matrix(matrix: PatternMatrix, column: int) -> PatternMatrix:
    """Retain wildcard/binder rows after removing one selected occurrence."""
    run_optional_validation(lambda: _validate_operation(matrix, column=column))
    selected = matrix.occurrences[column]
    rows = tuple(
        MatrixRow(
            cells=(*row.cells[:column], *row.cells[column + 1 :]),
            action_id=row.action_id,
            source_index=row.source_index,
            source_pattern_id=row.source_pattern_id,
            binder_assignments=_migrate_binder(row, wildcard, selected),
        )
        for row in matrix.rows
        if isinstance((wildcard := row.cells[column]), WildcardCell)
    )
    return PatternMatrix(
        occurrences=(*matrix.occurrences[:column], *matrix.occurrences[column + 1 :]),
        rows=rows,
        available_occurrences=matrix.available_occurrences,
        type_table=matrix.type_table,
        path_decompositions=matrix.path_decompositions,
        case_context=matrix.case_context,
    )


@dataclass(frozen=True, slots=True)
class QbaScore:
    """The three semantic qba score components for one refutable column."""

    leading_constructor_prefix: int
    distinct_branch_heads: int
    introduced_arity: int


@dataclass(frozen=True, slots=True)
class QbaSelection:
    """The deterministic qba choice and its inspectable score."""

    index: int
    occurrence: Occurrence
    score: QbaScore


def _qba_score(matrix: PatternMatrix, column: int) -> QbaScore:
    leading = 0
    for row in matrix.rows:
        if not isinstance(row.cells[column], ConstructorCell):
            break
        leading += 1
    heads = _head_constructors(matrix, column)
    return QbaScore(
        leading_constructor_prefix=leading,
        distinct_branch_heads=len(heads),
        introduced_arity=sum(head.arity for head in heads),
    )


def select_qba_column(matrix: PatternMatrix) -> QbaSelection:
    """Select a refutable column by q, then b, a, then occurrence order/ID."""
    run_optional_validation(lambda: _validate_operation(matrix))
    candidates: list[tuple[int, QbaScore]] = [
        (index, _qba_score(matrix, index))
        for index in range(len(matrix.occurrences))
        if any(isinstance(row.cells[index], ConstructorCell) for row in matrix.rows)
    ]
    if not candidates:
        raise MatchCompileInvariantError("qba selection requires at least one refutable column")

    def rank(candidate: tuple[int, QbaScore]) -> tuple[int, int, int, int, int, int]:
        candidate_index, candidate_score = candidate
        occurrence = matrix.occurrences[candidate_index]
        return (
            -candidate_score.leading_constructor_prefix,
            candidate_score.distinct_branch_heads,
            candidate_score.introduced_arity,
            occurrence.creation_order,
            occurrence.id.value,
            candidate_index,
        )

    index, score = min(
        candidates,
        key=rank,
    )
    return QbaSelection(index=index, occurrence=matrix.occurrences[index], score=score)


__all__ = [
    "OccurrenceAllocator",
    "PatternMatrix",
    "QbaScore",
    "QbaSelection",
    "Specialization",
    "default_matrix",
    "head_constructors",
    "matrix_from_normalized",
    "select_qba_column",
    "specialize",
    "validate_matrix",
]
