"""Memoized pattern-matrix compilation to immutable case-local decision DAGs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import EnumOwnerForm, EnumOwnerFormKind

from .diagnostics import (
    BoolWitness,
    EnumWitness,
    EnumWitnessQualification,
    LiteralWitness,
    MatchIssue,
    MatchWitness,
    NonExhaustiveIssue,
    OpenComplementWitness,
    RedundantArmIssue,
    WildcardWitness,
    WitnessField,
    issue_sort_key,
)
from .matrix import (
    OccurrenceAllocator,
    PatternMatrix,
    default_matrix,
    head_constructors,
    matrix_from_normalized,
    select_qba_column,
    specialize,
)
from .model import (
    BinderAssignment,
    BoolConstructor,
    ClosedSignature,
    Constructor,
    Decision,
    DecisionBranch,
    DecisionFail,
    DecisionLeaf,
    DecisionSwitch,
    EnumConstructor,
    EnumConstructorSpelling,
    FieldOccurrenceProvenance,
    LiteralConstructor,
    MatchCaseContext,
    MatrixRow,
    NormalizedCase,
    Occurrence,
    OccurrenceId,
    OpenSignature,
    SourceAction,
    WildcardCell,
)
from .normalize import MatchCompileInvariantError, signature_for_type


@dataclass(frozen=True, slots=True)
class CompiledCase:
    """One compiled source case, retained even when static issues are present."""

    normalized: NormalizedCase
    root: Decision
    occurrences: tuple[Occurrence, ...]
    reachable_action_ids: tuple[int, ...]
    issues: tuple[MatchIssue, ...]

    @property
    def case_node_id(self) -> int:
        """Stable source identity used as the later artifact mapping key."""
        return self.normalized.case_node_id

    @property
    def actions(self) -> tuple[SourceAction, ...]:
        """Source-priority action metadata consumed by diagnostics and lowering."""
        return self.normalized.actions


@dataclass(frozen=True, slots=True)
class _ConstructorConstraint:
    constructor: Constructor


@dataclass(frozen=True, slots=True)
class _OpenConstraint:
    excluded: tuple[LiteralConstructor, ...]


_Constraint: TypeAlias = _ConstructorConstraint | _OpenConstraint
_Constraints: TypeAlias = tuple[tuple[OccurrenceId, _Constraint], ...]


@dataclass(frozen=True, slots=True)
class _FailureAnalysis:
    """A deterministic failure suffix and the identities computed to derive it."""

    constraints: _Constraints | None
    analyzed_decision_ids: tuple[int, ...]


def _constructor_index(constructor: Constructor, signature: ClosedSignature) -> int:
    try:
        return signature.constructors.index(constructor)
    except ValueError as exc:
        raise MatchCompileInvariantError(
            "observed constructor is absent from its occurrence's closed signature"
        ) from exc


def _ordered_heads(matrix: PatternMatrix, column: int) -> tuple[Constructor, ...]:
    observed = head_constructors(matrix, column)
    signature = signature_for_type(matrix.occurrences[column].type, matrix.type_table)
    if isinstance(signature, OpenSignature):
        return observed

    def constructor_index(constructor: Constructor) -> int:
        return _constructor_index(constructor, signature)

    return tuple(sorted(observed, key=constructor_index))


def _signature_is_complete(observed: tuple[Constructor, ...], signature: ClosedSignature) -> bool:
    return len(observed) == len(signature.constructors) and all(
        constructor in observed for constructor in signature.constructors
    )


def _finalize_binders(matrix: PatternMatrix, row: MatrixRow) -> tuple[BinderAssignment, ...]:
    assignments = list(row.binder_assignments)
    for cell, occurrence in zip(row.cells, matrix.occurrences, strict=True):
        if not isinstance(cell, WildcardCell):
            raise MatchCompileInvariantError("leaf finalization requires an irrefutable first row")
        if cell.binder is not None:
            assignments.append(BinderAssignment(occurrence.id, cell.binder))

    available = {occurrence.id: occurrence for occurrence in matrix.available_occurrences}
    binder_ids: set[int] = set()
    for assignment in assignments:
        if assignment.occurrence not in available:
            raise MatchCompileInvariantError(
                f"leaf binder refers to unavailable occurrence {assignment.occurrence.value}"
            )
        if assignment.binder.node_id in binder_ids:
            raise MatchCompileInvariantError("leaf assigns the same source binder more than once")
        binder_ids.add(assignment.binder.node_id)

    def assignment_key(item: BinderAssignment) -> tuple[int, int, int]:
        return (
            available[item.occurrence].creation_order,
            item.occurrence.value,
            item.binder.node_id,
        )

    return tuple(sorted(assignments, key=assignment_key))


def _child_ids(
    occurrences: tuple[Occurrence, ...], parent: OccurrenceId, constructor: Constructor
) -> frozenset[OccurrenceId]:
    return frozenset(
        occurrence.id
        for occurrence in occurrences
        if isinstance(occurrence.provenance, FieldOccurrenceProvenance)
        and occurrence.provenance.parent == parent
        and occurrence.provenance.constructor == constructor
    )


def _switch_free_occurrences(
    occurrence: Occurrence,
    branches: tuple[DecisionBranch, ...],
    default: Decision | None,
    occurrences: tuple[Occurrence, ...],
) -> tuple[OccurrenceId, ...]:
    required = {occurrence.id}
    for branch in branches:
        required.update(
            set(branch.decision.free_occurrences)
            - _child_ids(occurrences, occurrence.id, branch.constructor)
        )
    if default is not None:
        required.update(default.free_occurrences)
    by_id = {item.id: item for item in occurrences}
    if any(identifier not in by_id for identifier in required):
        raise MatchCompileInvariantError("decision free interface names an unknown occurrence")

    def occurrence_key(identifier: OccurrenceId) -> tuple[int, int]:
        return by_id[identifier].creation_order, identifier.value

    return tuple(sorted(required, key=occurrence_key))


class _CaseCompiler:
    """Mutable tables scoped to exactly one source case compilation."""

    def __init__(self, normalized: NormalizedCase) -> None:
        self._normalized = normalized
        self._memo: dict[PatternMatrix, Decision] = {}
        self._interned: dict[Decision, Decision] = {}

    def intern(self, decision: Decision) -> Decision:
        existing = self._interned.get(decision)
        if existing is not None:
            return existing
        self._interned[decision] = decision
        return decision

    def compile(
        self, matrix: PatternMatrix, allocator: OccurrenceAllocator
    ) -> tuple[Decision, OccurrenceAllocator]:
        memoized = self._memo.get(matrix)
        if memoized is not None:
            return memoized, allocator
        if not matrix.rows:
            decision = self.intern(DecisionFail())
            self._memo[matrix] = decision
            return decision, allocator

        first = matrix.rows[0]
        if all(isinstance(cell, WildcardCell) for cell in first.cells):
            decision = self.intern(DecisionLeaf(first.action_id, _finalize_binders(matrix, first)))
            self._memo[matrix] = decision
            return decision, allocator

        selection = select_qba_column(matrix)
        heads = _ordered_heads(matrix, selection.index)
        branches: list[DecisionBranch] = []
        current_allocator = allocator
        for constructor in heads:
            specialized = specialize(matrix, selection.index, constructor, current_allocator)
            child, current_allocator = self.compile(specialized.matrix, specialized.allocator)
            branches.append(DecisionBranch(constructor, child))

        signature = signature_for_type(selection.occurrence.type, matrix.type_table)
        needs_default = isinstance(signature, OpenSignature) or not _signature_is_complete(
            heads, signature
        )
        default: Decision | None = None
        if needs_default:
            default, current_allocator = self.compile(
                default_matrix(matrix, selection.index), current_allocator
            )
        branch_tuple = tuple(branches)
        free_occurrences = _switch_free_occurrences(
            selection.occurrence,
            branch_tuple,
            default,
            current_allocator.occurrences,
        )
        decision = self.intern(
            DecisionSwitch(
                selection.occurrence,
                branch_tuple,
                default,
                free_occurrences,
            )
        )
        self._memo[matrix] = decision
        return decision, current_allocator


def _default_constraint(decision: DecisionSwitch, type_table: TypeTable) -> _Constraint:
    signature = signature_for_type(decision.occurrence.type, type_table)
    observed = tuple(branch.constructor for branch in decision.keyed_children)
    if isinstance(signature, ClosedSignature):
        missing = next(
            (constructor for constructor in signature.constructors if constructor not in observed),
            None,
        )
        if missing is None:
            raise MatchCompileInvariantError("complete closed switch unexpectedly has a default")
        return _ConstructorConstraint(missing)
    excluded: list[LiteralConstructor] = []
    for constructor in observed:
        if not isinstance(constructor, LiteralConstructor):
            raise MatchCompileInvariantError(
                "an open-domain switch contains a non-literal constructor"
            )
        excluded.append(constructor)
    return _OpenConstraint(tuple(excluded))


def _analyze_first_failure(root: Decision, type_table: TypeTable) -> _FailureAnalysis:
    memo: dict[int, _Constraints | None] = {}
    active: set[int] = set()
    analyzed: list[int] = []

    def visit(decision: Decision) -> _Constraints | None:
        identifier = id(decision)
        if identifier in active:
            raise MatchCompileInvariantError("decision graph contains a cycle")
        if identifier in memo:
            return memo[identifier]
        analyzed.append(identifier)
        if isinstance(decision, DecisionFail):
            memo[identifier] = ()
            return ()
        if isinstance(decision, DecisionLeaf):
            memo[identifier] = None
            return None
        active.add(identifier)
        try:
            signature = signature_for_type(decision.occurrence.type, type_table)
            edges: list[tuple[int, Decision, _Constraint]] = []
            for branch_index, branch in enumerate(decision.keyed_children):
                order = (
                    _constructor_index(branch.constructor, signature)
                    if isinstance(signature, ClosedSignature)
                    else branch_index
                )
                edges.append((order, branch.decision, _ConstructorConstraint(branch.constructor)))
            if decision.default is not None:
                constraint = _default_constraint(decision, type_table)
                order = (
                    _constructor_index(constraint.constructor, signature)
                    if isinstance(signature, ClosedSignature)
                    and isinstance(constraint, _ConstructorConstraint)
                    else len(edges)
                )
                edges.append((order, decision.default, constraint))

            def edge_order(edge: tuple[int, Decision, _Constraint]) -> int:
                return edge[0]

            for _, child, constraint in sorted(edges, key=edge_order):
                suffix = visit(child)
                if suffix is None:
                    continue
                if any(
                    occurrence_id == decision.occurrence.id
                    for occurrence_id, _ in suffix
                ):
                    raise MatchCompileInvariantError(
                        "a failure path tests an occurrence more than once"
                    )
                result = ((decision.occurrence.id, constraint), *suffix)
                memo[identifier] = result
                return result
            memo[identifier] = None
            return None
        finally:
            active.remove(identifier)

    constraints = visit(root)
    return _FailureAnalysis(constraints, tuple(analyzed))


def _first_failure_constraints(root: Decision, type_table: TypeTable) -> _Constraints | None:
    return _analyze_first_failure(root, type_table).constraints


def _witness_for_occurrence(
    occurrence: Occurrence,
    constraints: dict[OccurrenceId, _Constraint],
    occurrences: tuple[Occurrence, ...],
    case_context: MatchCaseContext,
) -> MatchWitness:
    constraint = constraints.get(occurrence.id)
    if constraint is None:
        return WildcardWitness()
    if isinstance(constraint, _OpenConstraint):
        return OpenComplementWitness(occurrence.type, constraint.excluded)
    constructor = constraint.constructor
    if isinstance(constructor, BoolConstructor):
        return BoolWitness(constructor.value)
    if isinstance(constructor, LiteralConstructor):
        return LiteralWitness(constructor.kind, constructor.value)
    spelling = _source_spelling(constructor, case_context)
    if spelling.owner_name is None and not spelling.bare:
        return WildcardWitness()
    children_by_index = {
        child.provenance.field_index: child
        for child in occurrences
        if isinstance(child.provenance, FieldOccurrenceProvenance)
        and child.provenance.parent == occurrence.id
        and child.provenance.constructor == constructor
    }
    fields = tuple(
        WitnessField(
            field.name,
            _witness_for_occurrence(
                children_by_index[index], constraints, occurrences, case_context
            )
            if index in children_by_index
            else WildcardWitness(),
        )
        for index, field in enumerate(constructor.fields)
    )
    qualification: EnumWitnessQualification | None = (
        None if spelling.bare else spelling
    )
    return EnumWitness(
        constructor.enum_type,
        constructor.variant,
        fields,
        qualification,
    )


def _source_spelling(
    constructor: EnumConstructor, case_context: MatchCaseContext
) -> EnumConstructorSpelling:
    """Select the shortest valid source owner for one concrete enum type."""
    enum_type = constructor.enum_type
    declaration_identity = (
        enum_type.module_id,
        enum_type.name,
        constructor.variant,
    )
    if declaration_identity in case_context.bare_enum_constructors:
        return EnumConstructorSpelling(None, None, bare=True)

    matches = tuple(
        form
        for form in case_context.enum_owner_forms
        if form.match(enum_type) is not None
    )
    if not matches:
        return EnumConstructorSpelling(None, None)

    def candidate_key(
        candidate: EnumOwnerForm,
    ) -> tuple[int, str, bool]:
        spelling = candidate
        assert spelling.owner_name is not None
        qualifier = spelling.module_qualifier
        if qualifier is None:
            return (
                2,
                spelling.owner_name,
                candidate.kind is not EnumOwnerFormKind.LOCAL,
            )
        if qualifier:
            text = f"{'.'.join(qualifier)}::{spelling.owner_name}"
            return (
                len(qualifier) + 2,
                text,
                candidate.kind is not EnumOwnerFormKind.QUALIFIED_IMPORT,
            )
        return 3, f"::{spelling.owner_name}", False

    return min(matches, key=candidate_key)


def _witness_for_root(
    root: Occurrence,
    constraints: dict[OccurrenceId, _Constraint],
    occurrences: tuple[Occurrence, ...],
    type_table: TypeTable,
    case_context: MatchCaseContext,
) -> MatchWitness:
    if root.id in constraints:
        return _witness_for_occurrence(root, constraints, occurrences, case_context)
    signature = signature_for_type(root.type, type_table)
    whole_domain: _Constraint
    if isinstance(signature, OpenSignature):
        whole_domain = _OpenConstraint(())
    else:
        whole_domain = _ConstructorConstraint(signature.constructors[0])
    return _witness_for_occurrence(
        root,
        {**constraints, root.id: whole_domain},
        occurrences,
        case_context,
    )


def _reachable_actions(root: Decision) -> set[int]:
    reachable: set[int] = set()
    seen: set[int] = set()

    def visit(decision: Decision) -> None:
        identifier = id(decision)
        if identifier in seen:
            return
        seen.add(identifier)
        if isinstance(decision, DecisionLeaf):
            reachable.add(decision.action_id)
        elif isinstance(decision, DecisionSwitch):
            for branch in decision.keyed_children:
                visit(branch.decision)
            if decision.default is not None:
                visit(decision.default)

    visit(root)
    return reachable


def _issues(
    normalized: NormalizedCase,
    root: Decision,
    occurrences: tuple[Occurrence, ...],
) -> tuple[tuple[int, ...], tuple[MatchIssue, ...]]:
    reachable = _reachable_actions(root)
    reachable_in_source_order = tuple(
        action.action_id for action in normalized.actions if action.action_id in reachable
    )
    issues: list[MatchIssue] = [
        RedundantArmIssue(normalized.case_node_id, action.action_id, action.pattern_span)
        for action in normalized.actions
        if action.action_id not in reachable
    ]
    constraints = _first_failure_constraints(root, normalized.type_table)
    if constraints is not None:
        constraint_map = dict(constraints)
        issues.append(
            NonExhaustiveIssue(
                normalized.case_node_id,
                normalized.span,
                _witness_for_root(
                    normalized.root,
                    constraint_map,
                    occurrences,
                    normalized.type_table,
                    normalized.case_context,
                ),
            )
        )
    return reachable_in_source_order, tuple(sorted(issues, key=issue_sort_key))


def validate_decision_dag(root: Decision) -> None:
    """Assert acyclicity, unique switch keys, and one test per occurrence per path."""
    visiting: set[int] = set()
    visited: set[int] = set()

    def acyclic(decision: Decision) -> None:
        identifier = id(decision)
        if identifier in visiting:
            raise MatchCompileInvariantError("decision graph contains a cycle")
        if identifier in visited:
            return
        visiting.add(identifier)
        if isinstance(decision, DecisionSwitch):
            constructors = tuple(branch.constructor for branch in decision.keyed_children)
            if not constructors or len(set(constructors)) != len(constructors):
                raise MatchCompileInvariantError(
                    "decision switch keys must be non-empty and unique"
                )
            if len(set(decision.free_occurrences)) != len(decision.free_occurrences):
                raise MatchCompileInvariantError(
                    "decision free occurrence interface has duplicates"
                )
            if decision.occurrence.id not in decision.free_occurrences:
                raise MatchCompileInvariantError("decision switch omits its tested free occurrence")
            for branch in decision.keyed_children:
                acyclic(branch.decision)
            if decision.default is not None:
                acyclic(decision.default)
        visiting.remove(identifier)
        visited.add(identifier)

    acyclic(root)
    states: set[tuple[int, frozenset[OccurrenceId]]] = set()

    def paths(decision: Decision, tested: frozenset[OccurrenceId]) -> None:
        state = (id(decision), tested)
        if state in states:
            return
        states.add(state)
        if not isinstance(decision, DecisionSwitch):
            return
        if decision.occurrence.id in tested:
            raise MatchCompileInvariantError(
                f"occurrence {decision.occurrence.id.value} is tested more than once on a path"
            )
        next_tested = tested | {decision.occurrence.id}
        for branch in decision.keyed_children:
            paths(branch.decision, next_tested)
        if decision.default is not None:
            paths(decision.default, next_tested)

    paths(root, frozenset())


def compile_case(normalized: NormalizedCase) -> CompiledCase:
    """Compile one normalized source case and derive all structured issues from its DAG."""
    compiler = _CaseCompiler(normalized)
    root, allocator = compiler.compile(
        matrix_from_normalized(normalized), OccurrenceAllocator.for_case(normalized)
    )
    validate_decision_dag(root)
    occurrences = allocator.occurrences
    reachable, issues = _issues(normalized, root, occurrences)
    return CompiledCase(normalized, root, occurrences, reachable, issues)


__all__ = ["CompiledCase", "compile_case", "validate_decision_dag"]
