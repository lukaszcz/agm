"""Memoized pattern-matrix compilation to immutable case-local decision DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    qualified_owner_name,
)
from .matrix import (
    OccurrenceAllocator,
    OccurrenceIndex,
    PatternMatrix,
    _binder_assignment_sort_key,
    _occurrence_sort_key,
    default_matrix,
    head_constructors,
    matrix_from_normalized,
    select_qba_column,
    specialize,
)
from .model import (
    BinderAssignment,
    BinderProvenance,
    BoolConstructor,
    ClosedSignature,
    Constructor,
    ConstructorCell,
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
from .normalize import (
    MatchCompileInvariantError,
    constructor_inhabits_type,
    signature_for_type,
)


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


def _constructor_index(constructor: Constructor, signature: ClosedSignature) -> int:
    index = signature.index_of(constructor)
    if index is None:
        raise MatchCompileInvariantError(
            "observed constructor is absent from its occurrence's closed signature"
        )
    return index


def _ordered_heads(matrix: PatternMatrix, column: int) -> tuple[Constructor, ...]:
    observed = head_constructors(matrix, column)
    signature = signature_for_type(matrix.occurrences[column].type, matrix.type_table)
    if isinstance(signature, OpenSignature):
        return observed

    def constructor_index(constructor: Constructor) -> int:
        return _constructor_index(constructor, signature)

    return tuple(sorted(observed, key=constructor_index))


def _signature_is_complete(observed: tuple[Constructor, ...], signature: ClosedSignature) -> bool:
    if len(observed) != len(signature.constructors):
        return False
    observed_set = frozenset(observed)
    return all(constructor in observed_set for constructor in signature.constructors)


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
        return _binder_assignment_sort_key(item, available)

    return tuple(sorted(assignments, key=assignment_key))


def _switch_free_occurrences(
    occurrence: Occurrence,
    branches: tuple[DecisionBranch, ...],
    default: Decision | None,
    index: OccurrenceIndex,
) -> tuple[OccurrenceId, ...]:
    required = {occurrence.id}
    for branch in branches:
        required.update(
            set(branch.decision.free_occurrences)
            - index.child_ids(occurrence.id, branch.constructor)
        )
    if default is not None:
        required.update(default.free_occurrences)
    by_id = index.by_id
    if any(identifier not in by_id for identifier in required):
        raise MatchCompileInvariantError("decision free interface names an unknown occurrence")

    def occurrence_key(identifier: OccurrenceId) -> tuple[int, int]:
        return _occurrence_sort_key(by_id[identifier])

    return tuple(sorted(required, key=occurrence_key))


class _CaseCompiler:
    """Mutable tables scoped to exactly one source case compilation."""

    def __init__(self) -> None:
        self._memo: dict[_CompileStateKey, Decision] = {}
        self._interned: dict[object, Decision] = {}

    def intern(self, decision: Decision) -> Decision:
        """Hash-cons a bottom-up decision without recursively hashing its DAG."""
        if isinstance(decision, DecisionFail):
            key: object = ("fail",)
        elif isinstance(decision, DecisionLeaf):
            key = ("leaf", decision.action_id, decision.binder_assignments)
        else:
            key = (
                "switch",
                decision.occurrence,
                tuple(
                    (branch.constructor, id(branch.decision)) for branch in decision.keyed_children
                ),
                None if decision.default is None else id(decision.default),
                decision.free_occurrences,
            )
        existing = self._interned.get(key)
        if existing is not None:
            return existing
        self._interned[key] = decision
        return decision

    def compile(
        self, matrix: PatternMatrix, allocator: OccurrenceAllocator
    ) -> tuple[Decision, OccurrenceAllocator]:
        state_key = _compile_state_key(matrix)
        memoized = self._memo.get(state_key)
        if memoized is not None:
            return memoized, allocator
        if not matrix.rows:
            decision = self.intern(DecisionFail())
            self._memo[state_key] = decision
            return decision, allocator

        first = matrix.rows[0]
        if all(isinstance(cell, WildcardCell) for cell in first.cells):
            decision = self.intern(DecisionLeaf(first.action_id, _finalize_binders(matrix, first)))
            self._memo[state_key] = decision
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
            current_allocator.index,
        )
        decision = self.intern(
            DecisionSwitch(
                selection.occurrence,
                branch_tuple,
                default,
                free_occurrences,
            )
        )
        self._memo[state_key] = decision
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


def _first_failure_constraints(root: Decision, type_table: TypeTable) -> _Constraints | None:
    """Return the deterministic first failure path's constraints, or ``None``.

    Every decision identity is analyzed at most once, so a hash-consed DAG whose
    paths are exponential in its node count still costs one visit per node.
    """
    memo: dict[int, _Constraints | None] = {}
    active: set[int] = set()

    def visit(decision: Decision) -> _Constraints | None:
        identifier = id(decision)
        if identifier in active:
            raise MatchCompileInvariantError("decision graph contains a cycle")
        if identifier in memo:
            return memo[identifier]
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
                if any(occurrence_id == decision.occurrence.id for occurrence_id, _ in suffix):
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

    return visit(root)


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
    if spelling.bare:
        qualification = None
    else:
        assert spelling.owner_name is not None
        qualification = EnumWitnessQualification(
            owner_name=spelling.owner_name,
            module_qualifier=spelling.module_qualifier,
            qualifier_anchored=spelling.qualifier_anchored,
        )
    return EnumWitness(
        constructor.enum_type,
        constructor.variant,
        fields,
        qualification,
    )


def _short_spelling_blocked(
    form: EnumOwnerForm, variant: str, case_context: MatchCaseContext
) -> bool:
    """Return whether a module route makes *form*'s short spelling ambiguous for *variant*.

    Only a ``LOCAL``/``OPEN_IMPORT`` form spells its owner bare as
    ``owner_name`` -- the same qualifier a same-named module route competes
    for -- so only those kinds consult ``blocked_enum_variants``.
    """
    if form.kind not in (EnumOwnerFormKind.LOCAL, EnumOwnerFormKind.OPEN_IMPORT):
        return False
    return variant in case_context.blocked_enum_variants.get((form.owner_name or "",), frozenset())


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
        and not _short_spelling_blocked(form, constructor.variant, case_context)
    )
    if not matches:
        return EnumConstructorSpelling(None, None)

    def candidate_key(
        candidate: EnumOwnerForm,
    ) -> tuple[int, str, bool]:
        assert candidate.owner_name is not None
        text = qualified_owner_name(
            candidate.owner_name,
            candidate.module_qualifier,
            anchored=candidate.qualifier_anchored,
        )
        return (
            len(text),
            text,
            candidate.kind is not EnumOwnerFormKind.LOCAL,
        )

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
    root_signature = signature_for_type(normalized.root.type, normalized.type_table)
    if constraints is not None and not (
        isinstance(root_signature, ClosedSignature) and not root_signature.constructors
    ):
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


def compile_case(normalized: NormalizedCase) -> CompiledCase:
    """Compile one normalized source case and derive all structured issues from its DAG.

    The decision DAG and its structured issues are the compiler's product.
    ``validate_compiled_case`` is the single self-check entry point over that
    product: it re-verifies the DAG as well as the ledger and the issues. The
    whole-program stage runs it once per compiled case — at whichever boundary
    that case reaches — when optional match-compilation validation is enabled
    (see :mod:`agm.agl.self_validation`).
    """
    compiler = _CaseCompiler()
    root, allocator = compiler.compile(
        matrix_from_normalized(normalized), OccurrenceAllocator.for_case(normalized)
    )
    occurrences = allocator.occurrences
    reachable, issues = _issues(normalized, root, occurrences)
    return CompiledCase(normalized, root, occurrences, reachable, issues)


# ---------------------------------------------------------------------------
# Optional self-validation
#
# Invariant self-checks that re-verify this module's own output.  They never
# change the compiler's result and run only when optional match-compilation
# validation is enabled (see ``agm.agl.self_validation``); the test harness
# turns them on so every compile in the suite is validated.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReplayFailRule:
    """The compiler rule expected for an empty canonical matrix state."""


@dataclass(frozen=True, slots=True)
class _ReplayLeafRule:
    """The exact leaf semantics expected for an irrefutable first row."""

    action_id: int
    binder_assignments: tuple[BinderAssignment, ...]


@dataclass(frozen=True, slots=True)
class _ReplaySwitchRule:
    """The exact local switch semantics expected for one refutable state."""

    occurrence: Occurrence
    heads: tuple[Constructor, ...]
    has_default: bool


_ReplayRule: TypeAlias = _ReplayFailRule | _ReplayLeafRule | _ReplaySwitchRule


@dataclass(frozen=True, slots=True)
class _CompileStateKey:
    """A structural matrix key whose recursive hash is computed once per matrix."""

    occurrences: tuple[Occurrence, ...]
    rows: tuple[MatrixRow, ...]
    cached_hash: int = field(compare=False)

    def __hash__(self) -> int:
        return self.cached_hash


def _compile_state_key(matrix: PatternMatrix) -> _CompileStateKey:
    """Return the live matrix state that determines all later compilation work."""
    return _CompileStateKey(matrix.occurrences, matrix.rows, matrix.compile_state_hash)


@dataclass(frozen=True, slots=True)
class _BinderPathStep:
    constructor: Constructor
    field_index: int


_BinderPath = tuple[_BinderPathStep, ...]


def _source_binder_paths(
    normalized: NormalizedCase,
) -> dict[int, dict[int, tuple[BinderProvenance, _BinderPath]]]:
    paths_by_action: dict[int, dict[int, tuple[BinderProvenance, _BinderPath]]] = {}

    def collect(
        cell: WildcardCell | ConstructorCell,
        path: _BinderPath,
        binders: dict[int, tuple[BinderProvenance, _BinderPath]],
    ) -> None:
        if isinstance(cell, WildcardCell):
            if cell.binder is not None:
                binders[cell.binder.node_id] = (cell.binder, path)
            return
        for field_index, argument in enumerate(cell.arguments):
            collect(
                argument,
                (*path, _BinderPathStep(cell.constructor, field_index)),
                binders,
            )

    for row in normalized.rows:
        binders: dict[int, tuple[BinderProvenance, _BinderPath]] = {}
        for cell in row.cells:
            collect(cell, (), binders)
        paths_by_action[row.action_id] = binders
    return paths_by_action


def _validate_occurrence_ledger(
    normalized: NormalizedCase,
    occurrences: tuple[Occurrence, ...],
) -> tuple[
    dict[OccurrenceId, Occurrence],
    dict[tuple[OccurrenceId, Constructor], tuple[Occurrence, ...]],
]:
    if not occurrences or occurrences[0] is not normalized.root:
        raise MatchCompileInvariantError(
            "compiled occurrence ledger must begin with the normalized root identity"
        )

    by_id: dict[OccurrenceId, Occurrence] = {}
    groups: dict[tuple[OccurrenceId, Constructor], dict[int, Occurrence]] = {}
    for index, occurrence in enumerate(occurrences):
        if occurrence.id.value != index or occurrence.creation_order != index:
            raise MatchCompileInvariantError(
                "compiled occurrence ledger must retain stable contiguous allocation order"
            )
        by_id[occurrence.id] = occurrence
        if occurrence is normalized.root:
            continue
        provenance = occurrence.provenance
        if not isinstance(provenance, FieldOccurrenceProvenance):
            raise MatchCompileInvariantError(
                "only the normalized root may have root occurrence provenance"
            )
        parent = by_id.get(provenance.parent)
        if parent is None:
            raise MatchCompileInvariantError(
                "field occurrence is not dominated by an earlier ledger parent"
            )
        signature = signature_for_type(parent.type, normalized.type_table)
        if not isinstance(signature, ClosedSignature):
            raise MatchCompileInvariantError(
                "field occurrence decomposes an occurrence with an open signature"
            )
        canonical = next(
            (
                constructor
                for constructor in signature.constructors
                if constructor == provenance.constructor
            ),
            None,
        )
        if not isinstance(canonical, EnumConstructor):
            raise MatchCompileInvariantError(
                "field occurrence constructor does not match its parent's checked signature"
            )
        if not 0 <= provenance.field_index < len(canonical.fields):
            raise MatchCompileInvariantError("field occurrence index is outside constructor arity")
        field = canonical.fields[provenance.field_index]
        if provenance.field_name != field.name or occurrence.type != field.type:
            raise MatchCompileInvariantError(
                "field occurrence does not match its declaration-order field"
            )
        group = groups.setdefault((parent.id, canonical), {})
        if provenance.field_index in group:
            raise MatchCompileInvariantError(
                "compiled occurrence ledger duplicates a constructor field identity"
            )
        group[provenance.field_index] = occurrence

    complete_groups: dict[tuple[OccurrenceId, Constructor], tuple[Occurrence, ...]] = {}
    for key, indexed_children in groups.items():
        constructor = key[1]
        assert isinstance(constructor, EnumConstructor)
        expected_indices = set(range(constructor.arity))
        if set(indexed_children) != expected_indices:
            raise MatchCompileInvariantError(
                "compiled occurrence ledger contains an incomplete constructor field group"
            )
        complete_groups[key] = tuple(indexed_children[index] for index in range(constructor.arity))
    return by_id, complete_groups


def _canonical_switch_constructor(
    constructor: Constructor,
    occurrence: Occurrence,
    type_table: TypeTable,
) -> Constructor:
    if not constructor_inhabits_type(constructor, occurrence.type):
        raise MatchCompileInvariantError(
            "decision switch key is incompatible with its tested occurrence"
        )
    signature = signature_for_type(occurrence.type, type_table)
    if isinstance(signature, OpenSignature):
        assert isinstance(constructor, LiteralConstructor)
        return constructor
    canonical = next(
        (candidate for candidate in signature.constructors if candidate == constructor),
        None,
    )
    if canonical is None:
        raise MatchCompileInvariantError("decision switch key is absent from its checked signature")
    return canonical


def _validate_decision_dataflow(
    root: Decision,
    root_available: frozenset[OccurrenceId],
    occurrence_groups: dict[tuple[OccurrenceId, Constructor], tuple[Occurrence, ...]] | None,
) -> None:
    """Validate path invariants with must/may summaries over a shared DAG.

    Available occurrences are a must fact (intersection at joins); tested
    occurrences are a may fact (union at joins).  These summaries retain the
    path-sensitive invariants without enumerating exponentially many paths.
    """
    states: dict[int, tuple[Decision, frozenset[OccurrenceId], frozenset[OccurrenceId]]] = {}
    worklist: list[int] = []

    def merge(
        decision: Decision,
        available: frozenset[OccurrenceId],
        tested: frozenset[OccurrenceId],
    ) -> None:
        identifier = id(decision)
        previous = states.get(identifier)
        if previous is not None:
            _, previous_available, previous_tested = previous
            available &= previous_available
            tested |= previous_tested
            if available == previous_available and tested == previous_tested:
                return
        states[identifier] = (decision, available, tested)
        worklist.append(identifier)

    merge(root, root_available, frozenset())
    while worklist:
        decision, available, tested = states[worklist.pop()]
        if not isinstance(decision, DecisionSwitch):
            continue
        next_tested = tested | {decision.occurrence.id}
        for branch in decision.keyed_children:
            child_available = available
            if occurrence_groups is not None:
                children = occurrence_groups.get((decision.occurrence.id, branch.constructor), ())
                child_available |= frozenset(child.id for child in children)
            merge(branch.decision, child_available, next_tested)
        if decision.default is not None:
            merge(decision.default, available, next_tested)

    for decision, available, tested in states.values():
        if occurrence_groups is not None and not set(decision.free_occurrences).issubset(available):
            raise MatchCompileInvariantError(
                "decision free occurrence interface is unavailable on an incoming path"
            )
        if isinstance(decision, DecisionSwitch) and decision.occurrence.id in tested:
            raise MatchCompileInvariantError(
                f"occurrence {decision.occurrence.id.value} is tested more than once on a path"
            )


def _validate_compiled_decisions(
    compiled: CompiledCase,
    occurrences_by_id: dict[OccurrenceId, Occurrence],
    occurrence_groups: dict[tuple[OccurrenceId, Constructor], tuple[Occurrence, ...]],
) -> None:
    normalized = compiled.normalized
    source_actions = {action.action_id: action for action in normalized.actions}
    binder_paths = _source_binder_paths(normalized)
    referenced_groups: set[tuple[OccurrenceId, Constructor]] = set()
    free_memo: dict[int, tuple[OccurrenceId, ...]] = {}
    occurrence_index = OccurrenceIndex.for_occurrences(compiled.occurrences)

    def resolve_binder_path(path: _BinderPath) -> Occurrence:
        occurrence = normalized.root
        for step in path:
            children = occurrence_groups[(occurrence.id, step.constructor)]
            occurrence = children[step.field_index]
        return occurrence

    def validate_free_interface(decision: Decision) -> tuple[OccurrenceId, ...]:
        identifier = id(decision)
        memoized = free_memo.get(identifier)
        if memoized is not None:
            return memoized
        if isinstance(decision, DecisionFail):
            expected_free: tuple[OccurrenceId, ...] = ()
        elif isinstance(decision, DecisionLeaf):
            if decision.action_id not in source_actions or decision.action_id not in binder_paths:
                raise MatchCompileInvariantError(
                    "decision leaf action does not belong to a retained source case row"
                )
            expected_binders = binder_paths[decision.action_id]
            actual_binders: dict[int, BinderAssignment] = {}
            for assignment in decision.binder_assignments:
                occurrence = occurrences_by_id.get(assignment.occurrence)
                if occurrence is None:
                    raise MatchCompileInvariantError(
                        "decision leaf binder names an unknown occurrence"
                    )
                if assignment.binder.node_id in actual_binders:
                    raise MatchCompileInvariantError(
                        "decision leaf assigns the same source binder more than once"
                    )
                expected = expected_binders.get(assignment.binder.node_id)
                if expected is None or assignment.binder != expected[0]:
                    raise MatchCompileInvariantError(
                        "decision leaf contains foreign source binder provenance"
                    )
                if occurrence is not resolve_binder_path(expected[1]):
                    raise MatchCompileInvariantError(
                        "decision leaf binder targets an incompatible occurrence identity"
                    )
                actual_binders[assignment.binder.node_id] = assignment
            if set(actual_binders) != set(expected_binders):
                raise MatchCompileInvariantError(
                    "decision leaf does not assign exactly its source action binders"
                )

            def assignment_order(item: BinderAssignment) -> tuple[int, int, int]:
                return _binder_assignment_sort_key(item, occurrences_by_id)

            expected_assignments = tuple(sorted(actual_binders.values(), key=assignment_order))
            if decision.binder_assignments != expected_assignments:
                raise MatchCompileInvariantError(
                    "decision leaf binder assignments are not in stable occurrence order"
                )
            expected_free = decision.free_occurrences
        else:
            ledger_occurrence = occurrences_by_id.get(decision.occurrence.id)
            if ledger_occurrence is not decision.occurrence:
                raise MatchCompileInvariantError(
                    "decision switch does not reference its exact ledger occurrence"
                )
            signature = signature_for_type(decision.occurrence.type, normalized.type_table)
            canonical_keys = tuple(
                _canonical_switch_constructor(
                    branch.constructor,
                    decision.occurrence,
                    normalized.type_table,
                )
                for branch in decision.keyed_children
            )
            if isinstance(signature, ClosedSignature):
                expected_order = tuple(
                    constructor
                    for constructor in signature.constructors
                    if constructor in canonical_keys
                )
                if canonical_keys != expected_order:
                    raise MatchCompileInvariantError(
                        "closed decision switch keys are not in signature order"
                    )
                complete = len(canonical_keys) == len(signature.constructors)
                if (decision.default is None) != complete:
                    raise MatchCompileInvariantError(
                        "closed decision switch default does not match signature coverage"
                    )
            elif decision.default is None:
                raise MatchCompileInvariantError(
                    "open-domain decision switch must retain a default"
                )

            for branch, canonical in zip(decision.keyed_children, canonical_keys, strict=True):
                if isinstance(canonical, EnumConstructor) and canonical.arity:
                    group_key = (decision.occurrence.id, canonical)
                    if group_key not in occurrence_groups:
                        raise MatchCompileInvariantError(
                            "enum decision branch lacks its complete occurrence field group"
                        )
                    referenced_groups.add(group_key)
                validate_free_interface(branch.decision)
            if decision.default is not None:
                validate_free_interface(decision.default)
            expected_free = _switch_free_occurrences(
                decision.occurrence,
                decision.keyed_children,
                decision.default,
                occurrence_index,
            )
            if decision.free_occurrences != expected_free:
                raise MatchCompileInvariantError(
                    "decision switch carries a forged free occurrence interface"
                )
        free_memo[identifier] = expected_free
        return expected_free

    _validate_decision_shape(compiled.root)
    validate_free_interface(compiled.root)
    if set(occurrence_groups) != referenced_groups:
        raise MatchCompileInvariantError(
            "compiled occurrence ledger does not exactly match decision decompositions"
        )

    # The ledger-aware dataflow subsumes the occurrence-free variant run by
    # ``validate_decision_dag``: it re-checks the one-test-per-occurrence-per-path
    # invariant over the same nodes, and additionally checks free interfaces
    # against the occurrences a path really makes available.
    _validate_decision_dataflow(
        compiled.root,
        frozenset((normalized.root.id,)),
        occurrence_groups,
    )


def _validate_semantic_replay(compiled: CompiledCase) -> None:
    """Replay compiler rules against the stored DAG without compiling a replacement.

    Canonical matrices memoize their exact decision identity, while decision identities
    memoize their local compiler rule. This accepts hash-consed nodes reached from
    semantically compatible states but rejects divergent state mappings.
    """

    state_nodes: dict[_CompileStateKey, Decision] = {}
    node_rules: dict[int, _ReplayRule] = {}

    def remember_rule(decision: Decision, rule: _ReplayRule) -> None:
        previous = node_rules.get(id(decision))
        if previous is not None and previous != rule:
            raise MatchCompileInvariantError(
                "semantic replay found one decision identity reused for incompatible states"
            )
        node_rules[id(decision)] = rule

    def replay(
        matrix: PatternMatrix,
        decision: Decision,
        allocator: OccurrenceAllocator,
    ) -> OccurrenceAllocator:
        state_key = _compile_state_key(matrix)
        if state_key in state_nodes:
            if state_nodes[state_key] is not decision:
                raise MatchCompileInvariantError(
                    "semantic replay found one canonical matrix state mapped to divergent "
                    "decision identities"
                )
            return allocator
        state_nodes[state_key] = decision

        if not matrix.rows:
            rule: _ReplayRule = _ReplayFailRule()
            remember_rule(decision, rule)
            if not isinstance(decision, DecisionFail):
                raise MatchCompileInvariantError(
                    "semantic replay requires an empty matrix to map to DecisionFail"
                )
            return allocator

        first = matrix.rows[0]
        if all(isinstance(cell, WildcardCell) for cell in first.cells):
            expected_assignments = _finalize_binders(matrix, first)
            rule = _ReplayLeafRule(first.action_id, expected_assignments)
            remember_rule(decision, rule)
            if not isinstance(decision, DecisionLeaf):
                raise MatchCompileInvariantError(
                    "semantic replay requires an irrefutable first row to map to DecisionLeaf"
                )
            if (
                decision.action_id != first.action_id
                or decision.binder_assignments != expected_assignments
            ):
                raise MatchCompileInvariantError(
                    "semantic replay found a leaf that does not select the canonical first row"
                )
            return allocator

        selection = select_qba_column(matrix)
        heads = _ordered_heads(matrix, selection.index)
        signature = signature_for_type(selection.occurrence.type, matrix.type_table)
        needs_default = isinstance(signature, OpenSignature) or not _signature_is_complete(
            heads, signature
        )
        rule = _ReplaySwitchRule(selection.occurrence, heads, needs_default)
        remember_rule(decision, rule)
        if not isinstance(decision, DecisionSwitch):
            raise MatchCompileInvariantError(
                "semantic replay requires a refutable matrix to map to DecisionSwitch"
            )
        if decision.occurrence != selection.occurrence:
            raise MatchCompileInvariantError(
                "semantic replay found a switch on the wrong qba-selected occurrence"
            )
        actual_heads = tuple(branch.constructor for branch in decision.keyed_children)
        if actual_heads != heads:
            raise MatchCompileInvariantError(
                "semantic replay found switch heads that differ from the observed matrix heads"
            )
        if (decision.default is not None) != needs_default:
            raise MatchCompileInvariantError(
                "semantic replay found a switch default inconsistent with its signature state"
            )
        current_allocator = allocator
        for branch, constructor in zip(decision.keyed_children, heads, strict=True):
            specialized = specialize(
                matrix,
                selection.index,
                constructor,
                current_allocator,
            )
            current_allocator = replay(
                specialized.matrix,
                branch.decision,
                specialized.allocator,
            )
        if needs_default:
            assert decision.default is not None
            current_allocator = replay(
                default_matrix(matrix, selection.index),
                decision.default,
                current_allocator,
            )
        return current_allocator

    final_allocator = replay(
        matrix_from_normalized(compiled.normalized),
        compiled.root,
        OccurrenceAllocator.for_case(compiled.normalized),
    )
    if final_allocator.occurrences != compiled.occurrences:
        raise MatchCompileInvariantError(
            "semantic replay occurrence allocation does not match the compiled ledger"
        )


def validate_compiled_case(
    compiled: CompiledCase,
    *,
    expected_normalized: NormalizedCase | None = None,
    require_success: bool = False,
) -> None:
    """Validate the complete compiler and source contract of one compiled case.

    ``expected_normalized`` lets an artifact boundary bind the case back to a
    freshly normalized checked source case. Invalid per-case compilation results
    remain inspectable unless ``require_success`` is requested.
    """
    normalized = compiled.normalized
    if expected_normalized is not None:
        if normalized != expected_normalized:
            raise MatchCompileInvariantError(
                "compiled case does not match its checked normalized source case"
            )
        if normalized.type_table is not expected_normalized.type_table:
            raise MatchCompileInvariantError(
                "compiled case belongs to a different checked type context"
            )
        actual_context = normalized.case_context
        expected_context = expected_normalized.case_context
        if (
            actual_context != expected_context
            or actual_context.owner_program is not expected_context.owner_program
        ):
            raise MatchCompileInvariantError(
                "compiled case belongs to a different checked source context"
            )

    matrix_from_normalized(normalized)
    occurrences_by_id, occurrence_groups = _validate_occurrence_ledger(
        normalized, compiled.occurrences
    )
    _validate_compiled_decisions(compiled, occurrences_by_id, occurrence_groups)
    _validate_semantic_replay(compiled)
    reachable, issues = _issues(normalized, compiled.root, compiled.occurrences)
    if compiled.reachable_action_ids != reachable:
        raise MatchCompileInvariantError(
            "compiled case reachable action ids do not match its decision DAG"
        )
    if compiled.issues != issues:
        raise MatchCompileInvariantError(
            "compiled case issues do not match authoritative decision-DAG analysis"
        )
    if require_success and issues:
        raise MatchCompileInvariantError(
            "successful match artifact contains issues from failure or redundant actions"
        )


def _validate_decision_shape(root: Decision) -> None:
    """Assert acyclicity and well-formed switch keys and free interfaces."""
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


def validate_decision_dag(root: Decision) -> None:
    """Assert acyclicity, unique switch keys, and one test per occurrence per path."""
    _validate_decision_shape(root)
    _validate_decision_dataflow(root, frozenset(), None)


__all__ = [
    "CompiledCase",
    "compile_case",
    "validate_compiled_case",
    "validate_decision_dag",
]
