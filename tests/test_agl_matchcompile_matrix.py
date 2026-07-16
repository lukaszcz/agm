"""Pattern-matrix decomposition and qba column-selection contracts."""

from __future__ import annotations

import decimal
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import agm.agl.matchcompile.matrix as matrix_module
from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.matchcompile.matrix import (
    OccurrenceAllocator,
    PatternMatrix,
    QbaScore,
    default_matrix,
    head_constructors,
    matrix_from_normalized,
    select_qba_column,
    specialize,
)
from agm.agl.matchcompile.model import (
    BinderAssignment,
    BoolConstructor,
    ConstructorCell,
    ConstructorField,
    EnumConstructor,
    FieldOccurrenceProvenance,
    LiteralConstructor,
    LiteralKind,
    MatrixRow,
    Occurrence,
    OccurrenceId,
    PathDecomposition,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import (
    MatchCompileInvariantError,
    normalize_case,
)
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import BoolType, EnumType, IntType, TextType
from agm.agl.semantics.values import BoolValue, EnumValue
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedModule, CheckedProgram, check, check_graph
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.match_reference import matrix_action, reference_action

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def _check(source: str) -> CheckedProgram:
    return check(resolve(parse_program(source)), _CAPS)


def _only_case(checked: CheckedProgram | CheckedModule) -> Case:
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert len(cases) == 1
    return cases[0]


def _pair_case() -> tuple[
    CheckedProgram, Case, PatternMatrix, EnumConstructor, OccurrenceAllocator
]:
    checked = _check(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "  | empty\n"
        "let subject: Pair = pair(left = false, right = true)\n"
        "case subject of\n"
        "  | pair(left = false, right = captured) => 1\n"
        "  | pair(left = true, right = false) => 2\n"
        "  | whole => 3\n"
    )
    case = _only_case(checked)
    normalized = normalize_case(case, checked)
    matrix = matrix_from_normalized(normalized)
    head = cast(ConstructorCell, matrix.rows[0].cells[0]).constructor
    assert isinstance(head, EnumConstructor)
    return checked, case, matrix, head, OccurrenceAllocator.for_case(normalized)


def test_specialization_and_default_are_exact_and_migrate_binders() -> None:
    _, case, matrix, pair, allocator = _pair_case()

    result = specialize(matrix, 0, pair, allocator)

    specialized = result.matrix
    assert [occurrence.id.value for occurrence in specialized.occurrences] == [1, 2]
    assert [occurrence.creation_order for occurrence in specialized.occurrences] == [1, 2]
    assert [occurrence.type for occurrence in specialized.occurrences] == [
        BoolType(),
        BoolType(),
    ]
    child_provenances = tuple(occurrence.provenance for occurrence in specialized.occurrences)
    assert all(
        isinstance(provenance, FieldOccurrenceProvenance) for provenance in child_provenances
    )
    assert [
        cast(FieldOccurrenceProvenance, provenance).field_name for provenance in child_provenances
    ] == ["left", "right"]
    assert [row.action_id for row in specialized.rows] == [
        branch.node_id for branch in case.branches
    ]
    first_left, first_right = specialized.rows[0].cells
    assert isinstance(first_left, ConstructorCell)
    assert first_left.constructor == BoolConstructor(False)
    assert isinstance(first_right, WildcardCell)
    assert first_right.binder is not None and first_right.binder.name == "captured"
    assert all(isinstance(cell, WildcardCell) for cell in specialized.rows[2].cells)
    whole = cast(WildcardCell, matrix.rows[2].cells[0]).binder
    assert whole is not None
    assert specialized.rows[2].binder_assignments == (
        BinderAssignment(matrix.occurrences[0].id, whole),
    )
    assert specialized.available_occurrences == (
        matrix.occurrences[0],
        *specialized.occurrences,
    )
    assert specialized.path_decompositions == (
        PathDecomposition(matrix.occurrences[0], pair, specialized.occurrences),
    )

    defaulted = default_matrix(matrix, 0)
    assert defaulted.occurrences == ()
    assert [row.action_id for row in defaulted.rows] == [case.branches[2].node_id]
    assert defaulted.rows[0].cells == ()
    assert defaulted.rows[0].binder_assignments == (
        BinderAssignment(matrix.occurrences[0].id, whole),
    )
    assert defaulted.available_occurrences == matrix.available_occurrences
    assert defaulted.path_decompositions == matrix.path_decompositions


def test_specialization_reuses_stable_children_and_allocates_sibling_ids_sequentially() -> None:
    checked = _check(
        "enum Choice\n"
        "  | first(value: bool)\n"
        "  | second(value: bool)\n"
        "let subject: Choice = first(value = true)\n"
        "case subject of | first(value = _) => 1 | second(value = _) => 2"
    )
    normalized = normalize_case(_only_case(checked), checked)
    matrix = matrix_from_normalized(normalized)
    first, second = head_constructors(matrix, 0)
    allocator = OccurrenceAllocator.for_case(normalized)

    first_result = specialize(matrix, 0, first, allocator)
    repeated = specialize(matrix, 0, first, first_result.allocator)
    second_result = specialize(matrix, 0, second, repeated.allocator)

    assert repeated.matrix.occurrences == first_result.matrix.occurrences
    assert repeated.allocator == first_result.allocator
    assert first_result.matrix.occurrences[0].id == OccurrenceId(1)
    assert second_result.matrix.occurrences[0].id == OccurrenceId(2)
    assert first_result.matrix.path_decompositions[0].children == (
        first_result.matrix.occurrences[0],
    )
    assert second_result.matrix.path_decompositions[0].children == (
        second_result.matrix.occurrences[0],
    )


def test_allocator_can_only_start_from_and_remains_bound_to_one_normalized_case() -> None:
    checked = _check(
        "enum Box\n"
        "  | box(value: bool)\n"
        "let subject = box(value = true)\n"
        "case subject of | box(value = true) => 1 | _ => 2"
    )
    case = _only_case(checked)
    normalized = normalize_case(case, checked)
    same_structure_other_case_context = normalize_case(case, checked)
    matrix = matrix_from_normalized(normalized)
    other_matrix = matrix_from_normalized(same_structure_other_case_context)
    box = head_constructors(matrix, 0)[0]
    allocator = OccurrenceAllocator.for_case(normalized)

    specialized = specialize(matrix, 0, box, allocator)
    repeated = specialize(matrix, 0, box, specialized.allocator)

    assert not hasattr(OccurrenceAllocator, "from_matrix")
    assert other_matrix == matrix
    assert hash(other_matrix) == hash(matrix)
    assert repeated.matrix.occurrences == specialized.matrix.occurrences
    assert repeated.allocator == specialized.allocator
    assert hash(repeated.allocator)
    with pytest.raises(MatchCompileInvariantError, match="normalized case root"):
        OccurrenceAllocator.for_case(specialized.matrix)
    with pytest.raises(MatchCompileInvariantError, match="case compilation"):
        specialize(other_matrix, 0, box, allocator)


def test_paper_decomposition_partition_preserves_first_match_actions() -> None:
    checked, case, matrix, pair, allocator = _pair_case()
    pair_result = specialize(matrix, 0, pair, allocator)
    defaulted = default_matrix(matrix, 0)
    enum_type = cast(EnumType, matrix.occurrences[0].type)
    nominal = NominalId(enum_type.module_id, enum_type.name)

    for left in (False, True):
        for right in (False, True):
            subject = EnumValue(
                nominal,
                enum_type.name,
                pair.variant,
                {"left": BoolValue(left), "right": BoolValue(right)},
            )
            assert matrix_action(
                pair_result.matrix, (BoolValue(left), BoolValue(right))
            ) == reference_action(case, checked, subject)

    empty = EnumValue(nominal, enum_type.name, "empty", {})
    assert matrix_action(defaulted, ()) == reference_action(case, checked, empty)


def test_qba_prefers_longest_leading_constructor_prefix() -> None:
    checked = _check(
        "enum Pair\n  | pair(left: bool, right: bool)\n"
        "let subject: Pair = pair(left = false, right = false)\n"
        "case subject of\n"
        "  | pair(left = false, right = false) => 1\n"
        "  | pair(left = true) => 2\n"
        "  | pair(right = true) => 3\n"
    )
    normalized = normalize_case(_only_case(checked), checked)
    outer = matrix_from_normalized(normalized)
    pair = head_constructors(outer, 0)[0]
    matrix = specialize(outer, 0, pair, OccurrenceAllocator.for_case(normalized)).matrix

    selection = select_qba_column(matrix)

    assert selection.index == 0
    assert selection.score.leading_constructor_prefix == 2


def test_qba_prefers_fewer_runtime_semantic_branch_heads() -> None:
    checked = _check(
        "enum Pair\n  | pair(left: decimal, right: bool)\n"
        "let subject: Pair = pair(left = 1, right = false)\n"
        "case subject of\n"
        "  | pair(left = 1, right = false) => 1\n"
        "  | pair(left = 1.0, right = true) => 2\n"
        "  | pair(left = 1, right = false) => 3\n"
    )
    normalized = normalize_case(_only_case(checked), checked)
    outer = matrix_from_normalized(normalized)
    pair = head_constructors(outer, 0)[0]
    matrix = specialize(outer, 0, pair, OccurrenceAllocator.for_case(normalized)).matrix

    selection = select_qba_column(matrix)

    assert selection.index == 0
    assert selection.score == QbaScore(
        leading_constructor_prefix=3,
        distinct_branch_heads=1,
        introduced_arity=0,
    )
    assert len(head_constructors(matrix, 0)) == 1


def test_qba_prefers_lower_total_introduced_arity() -> None:
    checked = _check(
        "enum Box\n  | boxed(value: bool)\n"
        "enum Mark\n  | marked\n"
        "enum Pair\n  | pair(left: Box, right: Mark)\n"
        "let subject: Pair = pair(left = boxed(value = true), right = marked)\n"
        "case subject of\n"
        "  | pair(left = boxed(value = true), right = marked) => 1\n"
        "  | pair(left = boxed(value = false), right = marked) => 2\n"
    )
    normalized = normalize_case(_only_case(checked), checked)
    outer = matrix_from_normalized(normalized)
    pair = head_constructors(outer, 0)[0]
    matrix = specialize(outer, 0, pair, OccurrenceAllocator.for_case(normalized)).matrix

    selection = select_qba_column(matrix)

    assert selection.index == 1
    assert selection.score.introduced_arity == 0


def test_qba_breaks_complete_ties_by_occurrence_creation_order_then_id() -> None:
    _, _, outer, pair, allocator = _pair_case()
    matrix = specialize(outer, 0, pair, allocator).matrix
    rows = tuple(
        replace(
            row,
            cells=(
                ConstructorCell(
                    BoolConstructor(False),
                    (),
                    cast(ConstructorCell, row.cells[0]).provenance,
                ),
                ConstructorCell(
                    BoolConstructor(False),
                    (),
                    cast(ConstructorCell, row.cells[0]).provenance,
                ),
            ),
        )
        for row in matrix.rows[:2]
    )
    tied = replace(matrix, rows=rows)
    assert select_qba_column(tied).index == 0

    right_occurrences = (
        replace(tied.occurrences[0], creation_order=8, id=OccurrenceId(8)),
        replace(tied.occurrences[1], creation_order=7, id=OccurrenceId(9)),
    )
    right_first = PatternMatrix(
        right_occurrences,
        tied.rows,
        (tied.available_occurrences[0], *right_occurrences),
        tied.type_table,
        (replace(tied.path_decompositions[0], children=right_occurrences),),
    )
    assert select_qba_column(right_first).index == 1

    lower_id_occurrences = (
        replace(tied.occurrences[0], creation_order=7, id=OccurrenceId(8)),
        replace(tied.occurrences[1], creation_order=7, id=OccurrenceId(9)),
    )
    lower_id_first = PatternMatrix(
        lower_id_occurrences,
        tied.rows,
        (tied.available_occurrences[0], *lower_id_occurrences),
        tied.type_table,
        (replace(tied.path_decompositions[0], children=lower_id_occurrences),),
    )
    assert select_qba_column(lower_id_first).index == 0


def test_qba_selection_validates_its_matrix_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, matrix, _, _ = _pair_case()
    calls = 0
    original = matrix_module.validate_matrix

    def count_validation(candidate: PatternMatrix) -> None:
        nonlocal calls
        calls += 1
        original(candidate)

    monkeypatch.setattr(matrix_module, "validate_matrix", count_validation)
    select_qba_column(matrix)
    assert calls == 1


def test_head_constructors_preserve_first_observation_order_and_semantic_uniqueness() -> None:
    checked = _check(
        "let left: decimal = 1\n"
        'let right: text = "x"\n'
        "case left of | 2 => 1 | 1 => 2 | 2.0 => 3 | _ => 4"
    )
    matrix = matrix_from_normalized(normalize_case(_only_case(checked), checked))
    assert head_constructors(matrix, 0) == (
        LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal("2")),
        LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal("1")),
    )


def test_matrix_operations_reject_malformed_boundaries_loudly() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    row = matrix.rows[0]

    with pytest.raises(MatchCompileInvariantError, match="row width"):
        PatternMatrix(
            matrix.occurrences,
            (replace(row, cells=()),),
            matrix.available_occurrences,
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="source order"):
        PatternMatrix(
            matrix.occurrences,
            (replace(matrix.rows[0], source_index=1), replace(matrix.rows[1], source_index=0)),
            matrix.available_occurrences,
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="available occurrence ids"):
        PatternMatrix(
            matrix.occurrences,
            matrix.rows,
            (matrix.occurrences[0], matrix.occurrences[0]),
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="active occurrence ids"):
        PatternMatrix(
            (matrix.occurrences[0], matrix.occurrences[0]),
            (),
            matrix.available_occurrences,
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="not available"):
        PatternMatrix(matrix.occurrences, matrix.rows, (), matrix.type_table)
    with pytest.raises(MatchCompileInvariantError, match="incompatible"):
        bad = ConstructorCell(
            LiteralConstructor(LiteralKind.TEXT, "bad"),
            (),
            cast(ConstructorCell, row.cells[0]).provenance,
        )
        PatternMatrix(
            matrix.occurrences,
            (replace(row, cells=(bad,)),),
            matrix.available_occurrences,
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="binder assignment"):
        binder = cast(WildcardCell, matrix.rows[2].cells[0]).binder
        assert binder is not None
        bad_row = replace(
            row,
            binder_assignments=(BinderAssignment(OccurrenceId(99), binder),),
        )
        PatternMatrix(
            matrix.occurrences,
            (bad_row,),
            matrix.available_occurrences,
            matrix.type_table,
        )
    with pytest.raises(MatchCompileInvariantError, match="column"):
        specialize(matrix, 2, pair, allocator)
    with pytest.raises(MatchCompileInvariantError, match="not observed"):
        specialize(matrix, 0, BoolConstructor(False), allocator)
    with pytest.raises(MatchCompileInvariantError, match="column"):
        default_matrix(matrix, -1)
    wildcard_only = default_matrix(matrix, 0)
    with pytest.raises(MatchCompileInvariantError, match="refutable"):
        select_qba_column(wildcard_only)


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("let value: int = 1\ncase value of | 1 => 1 | _ => 0", "1.5"),
        ("let value: decimal = 1\ncase value of | 1 => 1 | _ => 0", "NaN"),
        ("let value: decimal = 1\ncase value of | 1 => 1 | _ => 0", "Infinity"),
        ("let value: decimal = 1\ncase value of | 1 => 1 | _ => 0", "-Infinity"),
    ],
)
def test_matrix_rejects_manually_constructed_uninhabited_numeric_head(
    source: str, value: str
) -> None:
    checked = _check(source)
    matrix = matrix_from_normalized(normalize_case(_only_case(checked), checked))
    row = matrix.rows[0]
    original = cast(ConstructorCell, row.cells[0])
    impossible = replace(
        original,
        constructor=LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal(value)),
    )

    with pytest.raises(MatchCompileInvariantError, match="uninhabited"):
        replace(matrix, rows=(replace(row, cells=(impossible,)),))


def test_matrix_rejects_bad_constructor_children_and_occurrence_provenance() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    result = specialize(matrix, 0, pair, allocator)
    specialized = result.matrix
    first = cast(ConstructorCell, specialized.rows[0].cells[0])
    wrong_nested = ConstructorCell(
        EnumConstructor(
            enum_type=cast(EnumType, matrix.occurrences[0].type),
            variant=pair.variant,
            fields=(ConstructorField("wrong", BoolType()),),
        ),
        (WildcardCell(None, first.provenance),),
        first.provenance,
    )
    with pytest.raises(MatchCompileInvariantError, match="incompatible"):
        replace(
            specialized,
            rows=(replace(specialized.rows[0], cells=(wrong_nested, first)),),
        )

    bad_child = replace(
        specialized.occurrences[0],
        type=TextType(),
    )
    with pytest.raises(MatchCompileInvariantError, match="field occurrence"):
        bad_occurrences = (bad_child, specialized.occurrences[1])
        PatternMatrix(
            bad_occurrences,
            specialized.rows,
            (matrix.occurrences[0], *bad_occurrences),
            matrix.type_table,
            (replace(specialized.path_decompositions[0], children=bad_occurrences),),
        )


def test_matrix_rejects_uninhabited_constructor_nested_in_enum_field() -> None:
    checked = _check(
        "enum Box\n"
        "  | box(value: int)\n"
        "let subject: Box = box(value = 1)\n"
        "case subject of | box(value = 1) => 1 | _ => 0"
    )
    normalized = normalize_case(_only_case(checked), checked)
    matrix = matrix_from_normalized(normalized)
    row = matrix.rows[0]
    outer = cast(ConstructorCell, row.cells[0])
    nested = cast(ConstructorCell, outer.arguments[0])
    impossible = replace(
        nested,
        constructor=LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal("1.5")),
    )

    with pytest.raises(MatchCompileInvariantError, match="uninhabited"):
        replace(matrix, rows=(replace(row, cells=(replace(outer, arguments=(impossible,)),)),))


@pytest.mark.parametrize("defect", ["parent", "constructor", "index", "name", "type"])
def test_matrix_rejects_each_invalid_field_occurrence_provenance(defect: str) -> None:
    _, _, matrix, pair, allocator = _pair_case()
    specialized = specialize(matrix, 0, pair, allocator).matrix
    child = specialized.occurrences[0]
    provenance = cast(FieldOccurrenceProvenance, child.provenance)
    parent = matrix.occurrences[0]
    if defect == "parent":
        bad_provenance = replace(provenance, parent=OccurrenceId(99))
    elif defect == "constructor":
        bad_provenance = replace(provenance, constructor=BoolConstructor(False))
    elif defect == "index":
        bad_provenance = replace(provenance, field_index=99)
    elif defect == "name":
        bad_provenance = replace(provenance, field_name="wrong")
    else:
        bad_provenance = provenance
    bad_child = replace(
        child,
        provenance=bad_provenance,
        type=TextType() if defect == "type" else child.type,
    )
    children = (bad_child, specialized.occurrences[1])
    with pytest.raises(MatchCompileInvariantError, match="field occurrence"):
        PatternMatrix(
            (),
            (),
            (parent, *children),
            matrix.type_table,
            (replace(specialized.path_decompositions[0], children=children),),
        )


def test_matrix_rejects_decomposed_parent_remaining_active() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    specialized = specialize(matrix, 0, pair, allocator).matrix

    with pytest.raises(MatchCompileInvariantError, match="decomposed parent.*active"):
        PatternMatrix(
            (matrix.occurrences[0], *specialized.occurrences),
            (),
            specialized.available_occurrences,
            matrix.type_table,
            specialized.path_decompositions,
        )


@pytest.mark.parametrize("defect", ["missing", "reordered", "duplicated"])
def test_matrix_rejects_incomplete_or_unordered_decomposition_children(defect: str) -> None:
    _, _, matrix, pair, allocator = _pair_case()
    specialized = specialize(matrix, 0, pair, allocator).matrix
    left, right = specialized.occurrences
    children: tuple[Occurrence, ...]
    if defect == "missing":
        children = (left,)
    elif defect == "reordered":
        children = (right, left)
    else:
        children = (left, left)

    with pytest.raises(MatchCompileInvariantError, match="path decomposition"):
        replace(
            specialized,
            path_decompositions=(replace(specialized.path_decompositions[0], children=children),),
        )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("unavailable_parent", "parent is not exactly available"),
        ("repeated_parent", "tests parent.*more than once"),
        ("incompatible_constructor", "constructor is incompatible"),
        ("unavailable_child", "child is not exactly available"),
        ("unowned_child", "does not belong to a path decomposition"),
    ],
)
def test_matrix_rejects_decomposition_availability_and_ownership_defects(
    defect: str, message: str
) -> None:
    _, _, matrix, pair, allocator = _pair_case()
    specialized = specialize(matrix, 0, pair, allocator).matrix
    decomposition = specialized.path_decompositions[0]
    decompositions: tuple[PathDecomposition, ...]
    if defect == "unavailable_parent":
        decompositions = (
            replace(
                decomposition,
                parent=replace(decomposition.parent, creation_order=99),
            ),
        )
    elif defect == "repeated_parent":
        decompositions = (decomposition, decomposition)
    elif defect == "incompatible_constructor":
        decompositions = (replace(decomposition, constructor=BoolConstructor(False), children=()),)
    elif defect == "unavailable_child":
        decompositions = (
            replace(
                decomposition,
                children=(
                    replace(decomposition.children[0], creation_order=99),
                    decomposition.children[1],
                ),
            ),
        )
    else:
        decompositions = ()

    with pytest.raises(MatchCompileInvariantError, match=message):
        replace(specialized, path_decompositions=decompositions)


def test_matrix_tracks_and_canonicalizes_valid_nested_decomposition_order() -> None:
    checked = _check(
        "enum Inner\n"
        "  | boxed(value: bool)\n"
        "enum Outer\n"
        "  | wrapped(inner: Inner)\n"
        "let subject: Outer = wrapped(inner = boxed(value = true))\n"
        "case subject of | wrapped(inner = boxed(value = true)) => 1 | _ => 2"
    )
    normalized = normalize_case(_only_case(checked), checked)
    root = matrix_from_normalized(normalized)
    allocator = OccurrenceAllocator.for_case(normalized)
    wrapped = head_constructors(root, 0)[0]
    outer_result = specialize(root, 0, wrapped, allocator)
    boxed = head_constructors(outer_result.matrix, 0)[0]

    nested = specialize(outer_result.matrix, 0, boxed, outer_result.allocator).matrix

    assert tuple(decomposition.parent.id for decomposition in nested.path_decompositions) == (
        root.occurrences[0].id,
        outer_result.matrix.occurrences[0].id,
    )
    assert nested.path_decompositions[0].children == outer_result.matrix.occurrences
    assert nested.path_decompositions[1].children == nested.occurrences
    assert hash(nested)
    reordered = replace(
        nested,
        path_decompositions=tuple(reversed(nested.path_decompositions)),
    )
    assert reordered == nested
    assert hash(reordered) == hash(nested)

    late_root = replace(root.occurrences[0], creation_order=99)
    malformed_available = tuple(
        late_root if occurrence.id == late_root.id else occurrence
        for occurrence in nested.available_occurrences
    )
    malformed_decompositions = (
        replace(nested.path_decompositions[0], parent=late_root),
        nested.path_decompositions[1],
    )
    with pytest.raises(MatchCompileInvariantError, match="dominated"):
        replace(
            nested,
            available_occurrences=malformed_available,
            path_decompositions=malformed_decompositions,
        )


def test_allocator_rejects_incompatible_origin_and_rows_reject_duplicate_binders() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    alien = replace(matrix.occurrences[0], id=OccurrenceId(10), creation_order=10)
    alien_matrix = replace(
        matrix,
        occurrences=(alien,),
        available_occurrences=(alien,),
    )
    with pytest.raises(MatchCompileInvariantError, match="allocator"):
        specialize(alien_matrix, 0, pair, allocator)
    with pytest.raises(MatchCompileInvariantError, match="next id"):
        specialize(matrix, 0, pair, replace(allocator, next_id=0))
    with pytest.raises(MatchCompileInvariantError, match="creation order"):
        specialize(matrix, 0, pair, replace(allocator, next_creation_order=0))

    binder = cast(WildcardCell, matrix.rows[2].cells[0]).binder
    assert binder is not None
    duplicate = replace(
        matrix.rows[2],
        cells=(WildcardCell(binder, matrix.rows[2].cells[0].provenance),),
        binder_assignments=(BinderAssignment(matrix.occurrences[0].id, binder),),
    )
    with pytest.raises(MatchCompileInvariantError, match="binder"):
        PatternMatrix(
            matrix.occurrences,
            (duplicate,),
            matrix.available_occurrences,
            matrix.type_table,
        )


def test_invalid_manually_constructed_constructor_metadata_is_rejected() -> None:
    _, _, matrix, pair, _ = _pair_case()
    provenance = cast(ConstructorCell, matrix.rows[0].cells[0]).provenance
    duplicate_fields = EnumConstructor(
        pair.enum_type,
        pair.variant,
        (
            ConstructorField("left", BoolType()),
            ConstructorField("left", BoolType()),
        ),
    )
    cell = ConstructorCell(
        duplicate_fields,
        (WildcardCell(None, provenance), WildcardCell(None, provenance)),
        provenance,
    )
    with pytest.raises(MatchCompileInvariantError, match="checked signature"):
        PatternMatrix(
            matrix.occurrences,
            (MatrixRow((cell,), 1, 0, 1),),
            matrix.available_occurrences,
            matrix.type_table,
        )


def test_same_runtime_constructor_key_must_have_identical_field_metadata() -> None:
    _, _, matrix, pair, _ = _pair_case()
    first = cast(ConstructorCell, matrix.rows[0].cells[0])
    reversed_constructor = EnumConstructor(
        pair.enum_type,
        pair.variant,
        tuple(reversed(pair.fields)),
    )
    reversed_cell = ConstructorCell(
        reversed_constructor,
        tuple(reversed(first.arguments)),
        first.provenance,
    )
    with pytest.raises(MatchCompileInvariantError, match="checked signature"):
        PatternMatrix(
            matrix.occurrences,
            (matrix.rows[0], replace(matrix.rows[1], cells=(reversed_cell,))),
            matrix.available_occurrences,
            matrix.type_table,
        )


def test_null_constructor_compatibility_and_nullary_specialization() -> None:
    null_checked = _check("let value: json = null\ncase value of | null => 1 | _ => 2")
    null_normalized = normalize_case(_only_case(null_checked), null_checked)
    null_matrix = matrix_from_normalized(null_normalized)
    null_head = head_constructors(null_matrix, 0)[0]
    assert (
        specialize(
            null_matrix,
            0,
            null_head,
            OccurrenceAllocator.for_case(null_normalized),
        ).matrix.occurrences
        == ()
    )

    bool_checked = _check("let value = true\ncase value of | true => 1 | _ => 2")
    bool_normalized = normalize_case(_only_case(bool_checked), bool_checked)
    bool_matrix = matrix_from_normalized(bool_normalized)
    bool_head = head_constructors(bool_matrix, 0)[0]
    specialized = specialize(
        bool_matrix,
        0,
        bool_head,
        OccurrenceAllocator.for_case(bool_normalized),
    ).matrix
    assert specialized.occurrences == ()
    assert specialized.rows[1].binder_assignments == ()


def test_specialization_rejects_same_key_with_different_metadata() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    incompatible_metadata = EnumConstructor(
        pair.enum_type,
        pair.variant,
        tuple(reversed(pair.fields)),
    )
    with pytest.raises(MatchCompileInvariantError, match="checked signature"):
        specialize(
            matrix,
            0,
            incompatible_metadata,
            allocator,
        )


@pytest.mark.parametrize(
    "defect",
    ["unknown_variant", "omitted", "invented", "reordered", "wrong_generic_type"],
)
def test_matrix_rejects_single_enum_head_that_disagrees_with_checked_signature(
    defect: str,
) -> None:
    checked = _check(
        "enum Pair[T]\n"
        "  | pair(left: T, right: bool)\n"
        "  | empty\n"
        'let subject: Pair[text] = pair(left = "x", right = true)\n'
        "case subject of | pair(left = _, right = _) => 1 | _ => 2"
    )
    matrix = matrix_from_normalized(normalize_case(_only_case(checked), checked))
    cell = cast(ConstructorCell, matrix.rows[0].cells[0])
    constructor = cast(EnumConstructor, cell.constructor)
    arguments = cell.arguments
    if defect == "unknown_variant":
        malformed = EnumConstructor(constructor.enum_type, "missing", ())
        arguments = ()
    elif defect == "omitted":
        malformed = replace(constructor, fields=constructor.fields[:1])
        arguments = arguments[:1]
    elif defect == "invented":
        malformed = replace(
            constructor,
            fields=(*constructor.fields, ConstructorField("extra", BoolType())),
        )
        arguments = (*arguments, WildcardCell(None, cell.provenance))
    elif defect == "reordered":
        malformed = replace(constructor, fields=tuple(reversed(constructor.fields)))
        arguments = tuple(reversed(arguments))
    else:
        malformed = replace(
            constructor,
            fields=(replace(constructor.fields[0], type=BoolType()), constructor.fields[1]),
        )
    malformed_cell = ConstructorCell(malformed, arguments, cell.provenance)

    with pytest.raises(MatchCompileInvariantError, match="checked signature"):
        replace(matrix, rows=(replace(matrix.rows[0], cells=(malformed_cell,)),))


def test_matrix_context_is_identity_only_and_mixed_allocators_are_rejected() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    _, _, other_matrix, _, other_allocator = _pair_case()

    assert matrix.type_table is not other_matrix.type_table
    same_semantics_other_context = replace(matrix, type_table=other_matrix.type_table)
    assert same_semantics_other_context == matrix
    assert hash(same_semantics_other_context) == hash(matrix)
    assert "type_table" not in repr(matrix)

    with pytest.raises(MatchCompileInvariantError, match="compiler context"):
        specialize(same_semantics_other_context, 0, pair, allocator)

    with pytest.raises(MatchCompileInvariantError, match="case compilation"):
        specialize(matrix, 0, pair, other_allocator)

    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        replace(matrix, type_table=TypeTable())


def test_path_decomposition_constructor_must_match_checked_signature() -> None:
    _, _, matrix, pair, allocator = _pair_case()
    specialized = specialize(matrix, 0, pair, allocator).matrix
    malformed = replace(pair, fields=tuple(reversed(pair.fields)))

    with pytest.raises(MatchCompileInvariantError, match="checked signature"):
        replace(
            specialized,
            path_decompositions=(
                replace(specialized.path_decompositions[0], constructor=malformed),
            ),
        )


def test_imported_generic_signature_is_canonical_during_matrix_specialization(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "lib": "enum Choice[T]\n  | absent\n  | present(value: T, note: text)",
            "entry": (
                "import lib\n"
                'let value: Choice[int] = present(value = 1, note = "x")\n'
                "case value of | present(value = _) => 1 | absent => 0"
            ),
        },
    )
    checked = check_graph(resolve_graph(graph), _CAPS).modules[ENTRY_ID]
    normalized = normalize_case(_only_case(checked), checked)
    matrix = matrix_from_normalized(normalized)
    head = head_constructors(matrix, 0)[0]

    specialized = specialize(matrix, 0, head, OccurrenceAllocator.for_case(normalized)).matrix

    assert [occurrence.type for occurrence in specialized.occurrences] == [
        IntType(),
        TextType(),
    ]


def _independent_box_columns() -> tuple[
    PatternMatrix,
    EnumConstructor,
    OccurrenceAllocator,
]:
    checked = _check(
        "enum Box\n"
        "  | boxed(value: bool)\n"
        "enum Pair\n"
        "  | pair(left: Box, right: Box)\n"
        "let subject = pair(left = boxed(value = true), right = boxed(value = false))\n"
        "case subject of\n"
        "  | pair(left = boxed(value = true), right = boxed(value = false)) => 1\n"
        "  | pair(left = boxed(value = left), right = boxed(value = right)) => 2\n"
        "  | pair(left = left_box, right = right_box) => 3\n"
    )
    normalized = normalize_case(_only_case(checked), checked)
    root = matrix_from_normalized(normalized)
    pair = head_constructors(root, 0)[0]
    pair_result = specialize(root, 0, pair, OccurrenceAllocator.for_case(normalized))
    boxed = head_constructors(pair_result.matrix, 0)[0]
    assert isinstance(boxed, EnumConstructor)
    return pair_result.matrix, boxed, pair_result.allocator


def _decompose_independent_columns(
    matrix: PatternMatrix,
    boxed: EnumConstructor,
    allocator: OccurrenceAllocator,
    order: tuple[int, int],
) -> tuple[PatternMatrix, OccurrenceAllocator]:
    current = matrix
    current_allocator = allocator
    for original_column in order:
        selected_id = matrix.occurrences[original_column].id
        column = next(
            index
            for index, occurrence in enumerate(current.occurrences)
            if occurrence.id == selected_id
        )
        result = specialize(current, column, boxed, current_allocator)
        current = result.matrix
        current_allocator = result.allocator
    return current, current_allocator


def test_independent_decomposition_orders_have_one_canonical_matrix_key() -> None:
    matrix, boxed, allocator = _independent_box_columns()
    left_then_right, complete_ledger = _decompose_independent_columns(
        matrix, boxed, allocator, (0, 1)
    )
    right_then_left, _ = _decompose_independent_columns(
        matrix, boxed, complete_ledger, (1, 0)
    )

    assert left_then_right == right_then_left
    assert hash(left_then_right) == hash(right_then_left)
    assert tuple(occurrence.id for occurrence in left_then_right.occurrences) == tuple(
        occurrence.id for occurrence in right_then_left.occurrences
    )
    assert tuple(len(row.binder_assignments) for row in left_then_right.rows) == (0, 0, 2)


def test_nested_decompositions_are_canonicalized_in_topological_order() -> None:
    matrix, boxed, allocator = _independent_box_columns()
    boxes, complete_ledger = _decompose_independent_columns(matrix, boxed, allocator, (0, 1))
    left_head = head_constructors(boxes, 0)[0]
    right_head = head_constructors(boxes, 1)[0]
    left_then_right = specialize(boxes, 0, left_head, complete_ledger)
    left_then_right = specialize(
        left_then_right.matrix,
        0,
        right_head,
        left_then_right.allocator,
    )

    right_first = specialize(boxes, 1, right_head, left_then_right.allocator)
    right_then_left = specialize(
        right_first.matrix,
        0,
        left_head,
        right_first.allocator,
    ).matrix

    assert left_then_right.matrix == right_then_left
    assert hash(left_then_right.matrix) == hash(right_then_left)
    decomposition_parent_orders = tuple(
        decomposition.parent.creation_order
        for decomposition in right_then_left.path_decompositions
    )
    assert decomposition_parent_orders == tuple(sorted(decomposition_parent_orders))


def test_manual_order_insensitive_state_is_canonicalized_without_erasing_semantics() -> None:
    matrix, boxed, allocator = _independent_box_columns()
    canonical, _ = _decompose_independent_columns(matrix, boxed, allocator, (0, 1))
    reordered = replace(
        canonical,
        rows=tuple(
            replace(row, binder_assignments=tuple(reversed(row.binder_assignments)))
            for row in canonical.rows
        ),
        available_occurrences=tuple(reversed(canonical.available_occurrences)),
        path_decompositions=tuple(reversed(canonical.path_decompositions)),
    )

    assert reordered == canonical
    assert hash(reordered) == hash(canonical)

    reversed_active = replace(
        canonical,
        occurrences=tuple(reversed(canonical.occurrences)),
        rows=tuple(replace(row, cells=tuple(reversed(row.cells))) for row in canonical.rows),
    )
    assert reversed_active != canonical

    binder_row = canonical.rows[2]
    left_assignment, right_assignment = binder_row.binder_assignments
    different_environment = replace(
        canonical,
        rows=(
            *canonical.rows[:2],
            replace(
                binder_row,
                binder_assignments=(
                    replace(left_assignment, occurrence=right_assignment.occurrence),
                    replace(right_assignment, occurrence=left_assignment.occurrence),
                ),
            ),
        ),
    )
    assert different_environment != canonical
