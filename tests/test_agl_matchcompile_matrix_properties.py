"""Paper-derived and semantic partition properties for pattern matrices."""

from __future__ import annotations

import decimal
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.matchcompile.compiler import compile_match_site, validate_compiled_case
from agm.agl.matchcompile.matrix import (
    OccurrenceAllocator,
    PatternMatrix,
    default_matrix,
    head_constructors,
    matrix_from_normalized,
    specialize,
)
from agm.agl.matchcompile.model import (
    BinderAssignment,
    BinderProvenance,
    Constructor,
    ConstructorCell,
    DecisionDecompose,
    MatrixRow,
    NominalConstructor,
    PatternCell,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import normalize_case
from agm.agl.semantics.types import EnumType
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    IntValue,
    JsonValue,
    RecordValue,
    TextValue,
    Value,
)
from agm.agl.syntax.nodes import Case, VarPattern
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedModule
from tests.agl.match_reference import (
    canonical_cell_matches,
    enum_variant_members,
    matrix_action,
    reference_action,
)
from tests.agl.module_graph import resolve_and_check_inline_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def _matrix(
    source: str,
) -> tuple[CheckedModule, Case, PatternMatrix, OccurrenceAllocator]:
    checked = resolve_and_check_inline_entry(source, _CAPS)
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert len(cases) == 1
    case = cases[0]
    normalized = normalize_case(case, checked)
    return (
        checked,
        case,
        matrix_from_normalized(normalized),
        OccurrenceAllocator.for_case(normalized),
    )


def _constructor_by_variant(matrix: PatternMatrix, column: int) -> dict[str, NominalConstructor]:
    constructors = head_constructors(matrix, column)
    assert all(isinstance(constructor, NominalConstructor) for constructor in constructors)
    return {
        constructor.terminal_name: constructor
        for constructor in cast(tuple[NominalConstructor, ...], constructors)
    }


def _source_binder(case: Case, source_index: int, name: str) -> BinderProvenance:
    """Build the provenance of a source branch's *name* binder from the AST alone."""
    binders: list[VarPattern] = []

    def collect(node: object) -> None:
        if isinstance(node, VarPattern) and node.name == name:
            binders.append(node)

    walk(case.branches[source_index].pattern, collect)
    assert len(binders) == 1
    binder = binders[0]
    return BinderProvenance(node_id=binder.node_id, name=binder.name, span=binder.span)


def _expected_row(
    case: Case,
    source_index: int,
    cells: tuple[PatternCell, ...],
    binder_assignments: tuple[BinderAssignment, ...] = (),
) -> MatrixRow:
    """Build an expected row whose identity comes from the source branch, not the matrix.

    Deriving ``action_id``/``source_index``/``source_pattern_id`` from the case
    AST keeps the expectation independent of the rows under test, so a
    decomposition that consistently corrupts a row identity cannot satisfy it.
    """
    branch = case.branches[source_index]
    return MatrixRow(
        cells=cells,
        action_id=branch.node_id,
        source_index=source_index,
        source_pattern_id=branch.pattern.node_id,
        binder_assignments=binder_assignments,
    )


def test_paper_specializations_preserve_complete_rows_and_priority() -> None:
    """Adapt Maranget's P, S(::, P), and S([], P) matrices to AgL enums."""
    _, case, root, allocator = _matrix(
        "enum List\n"
        "  | nil\n"
        "  | cons(head: int, tail: List)\n"
        "enum Subject\n"
        "  | subject(left: List, right: List)\n"
        "let value = subject(left = nil(), right = nil())\n"
        "case value of\n"
        "  | subject(left = nil(), right = _) => 1\n"
        "  | subject(left = left, right = nil()) => 2\n"
        "  | subject(left = cons(), right = cons()) => 3\n"
    )
    subject = cast(NominalConstructor, head_constructors(root, 0)[0])
    columns_result = specialize(root, 0, subject, allocator)
    columns = columns_result.matrix
    nil = _constructor_by_variant(columns, 0)["nil"]
    cons = _constructor_by_variant(columns, 0)["cons"]
    nil_result = specialize(columns, 0, nil, columns_result.allocator)
    cons_result = specialize(columns, 0, cons, nil_result.allocator)
    row_nil, row_wildcard, row_cons = columns.rows
    wildcard = cast(WildcardCell, row_wildcard.cells[0])
    explicit_cons = cast(ConstructorCell, row_cons.cells[0])
    # The second branch binds ``left``; specializing away column 0 must migrate
    # that binder onto the occurrence the column stood for.
    migrated_left = (BinderAssignment(columns.occurrences[0].id, _source_binder(case, 1, "left")),)

    expected_nil = (
        _expected_row(case, 0, (row_nil.cells[1],)),
        _expected_row(case, 1, (row_wildcard.cells[1],), migrated_left),
    )
    expected_cons = (
        _expected_row(
            case,
            1,
            (
                WildcardCell(wildcard.provenance),
                WildcardCell(wildcard.provenance),
                row_wildcard.cells[1],
            ),
            migrated_left,
        ),
        _expected_row(case, 2, (*explicit_cons.arguments, row_cons.cells[1])),
    )
    expected_default = (_expected_row(case, 1, (row_wildcard.cells[1],), migrated_left),)

    decompositions = (
        ("S(nil, P)", nil_result.matrix.rows, expected_nil),
        ("S(cons, P)", cons_result.matrix.rows, expected_cons),
        ("D(P)", default_matrix(columns, 0).rows, expected_default),
    )
    for name, actual, expected in decompositions:
        assert actual == expected, name


def test_paper_default_retains_and_migrates_all_wildcard_rows() -> None:
    """Adapt Maranget's Q and D(Q), retaining both wildcard-leading rows."""
    _, case, root, allocator = _matrix(
        "enum List\n"
        "  | nil\n"
        "  | cons(head: int, tail: List)\n"
        "enum Subject\n"
        "  | subject(left: List, right: List)\n"
        "let value = subject(left = nil(), right = nil())\n"
        "case value of\n"
        "  | subject(left = nil(), right = _) => 1\n"
        "  | subject(left = left, right = nil()) => 2\n"
        "  | subject(left = left, right = _) => 3\n"
    )
    subject = cast(NominalConstructor, head_constructors(root, 0)[0])
    matrix = specialize(root, 0, subject, allocator).matrix
    defaulted = default_matrix(matrix, 0)
    _, second, third = matrix.rows
    dropped_column = matrix.occurrences[0].id

    assert defaulted.rows == (
        _expected_row(
            case,
            1,
            (second.cells[1],),
            (BinderAssignment(dropped_column, _source_binder(case, 1, "left")),),
        ),
        _expected_row(
            case,
            2,
            (third.cells[1],),
            (BinderAssignment(dropped_column, _source_binder(case, 2, "left")),),
        ),
    )


def _head_arguments(
    constructor: Constructor,
    provenance_cell: ConstructorCell,
    value: Value,
    enum_variant_members: dict[tuple[NominalId, str], NominalId],
) -> tuple[Value, ...] | None:
    head_only = ConstructorCell(
        constructor,
        tuple(WildcardCell(provenance_cell.provenance) for _ in range(constructor.arity)),
        provenance_cell.provenance,
    )
    if not canonical_cell_matches(head_only, value, enum_variant_members):
        return None
    if isinstance(constructor, NominalConstructor):
        assert isinstance(value, RecordValue)
        return tuple(value.fields[field.name] for field in constructor.fields)
    return ()


def _assert_decomposition_partition(
    checked: CheckedModule,
    case: Case,
    matrix: PatternMatrix,
    allocator: OccurrenceAllocator,
    subjects: tuple[Value, ...],
) -> None:
    heads = head_constructors(matrix, 0)
    specialized: list[tuple[Constructor, PatternMatrix, ConstructorCell]] = []
    for head in heads:
        result = specialize(matrix, 0, head, allocator)
        allocator = result.allocator
        provenance_cell = next(
            row.cells[0]
            for row in matrix.rows
            if isinstance(row.cells[0], ConstructorCell) and row.cells[0].constructor == head
        )
        specialized.append((head, result.matrix, provenance_cell))
    defaulted = default_matrix(matrix, 0)
    members_by_variant = enum_variant_members(matrix.occurrences, matrix.type_table)

    for subject in subjects:
        expected = reference_action(case, checked, subject)
        matching = [
            (specialized_matrix, arguments)
            for head, specialized_matrix, provenance_cell in specialized
            if (arguments := _head_arguments(head, provenance_cell, subject, members_by_variant))
            is not None
        ]
        if matching:
            assert len(matching) == 1
            specialized_matrix, arguments = matching[0]
            assert matrix_action(specialized_matrix, arguments) == expected
        else:
            assert matrix_action(defaulted, ()) == expected


def test_boolean_and_enum_decompositions_partition_complete_finite_domains() -> None:
    bool_checked, bool_case, bool_matrix, bool_allocator = _matrix(
        "let value = false\ncase value of | false => 1 | _ as remaining => 2"
    )
    _assert_decomposition_partition(
        bool_checked,
        bool_case,
        bool_matrix,
        bool_allocator,
        (BoolValue(False), BoolValue(True)),
    )

    enum_checked, enum_case, enum_matrix, enum_allocator = _matrix(
        "enum Color\n"
        "  | red\n"
        "  | green\n"
        "  | blue\n"
        "let value: Color = red()\n"
        "case value of | red() => 1 | blue() => 2 | _ as remaining => 3"
    )
    enum_type = cast(EnumType, enum_matrix.occurrences[0].type)
    nominal = NominalId(enum_type.decl_id)
    subjects = tuple(
        RecordValue(nominal=nominal, display_name=f"{enum_type.name}::{variant}", fields={})
        for variant in ("red", "green", "blue")
    )
    _assert_decomposition_partition(enum_checked, enum_case, enum_matrix, enum_allocator, subjects)


@pytest.mark.parametrize(
    ("source", "subjects"),
    [
        (
            "let value: decimal = 1\ncase value of | 1 => 1 | 2.5 => 2 | _ => 3",
            (
                IntValue(1),
                DecimalValue(decimal.Decimal("1.0")),
                DecimalValue(decimal.Decimal("2.5")),
                DecimalValue(decimal.Decimal("9")),
            ),
        ),
        (
            'let value = "x"\ncase value of | "x" => 1 | "y" => 2 | _ => 3',
            (TextValue("x"), TextValue("y"), TextValue("other")),
        ),
        (
            "let value: json = null\ncase value of | null => 1 | _ => 2",
            (JsonValue(None), JsonValue("not null"), JsonValue(1)),
        ),
    ],
)
def test_scalar_decompositions_use_runtime_literal_equality(
    source: str,
    subjects: tuple[Value, ...],
) -> None:
    checked, case, matrix, allocator = _matrix(source)
    _assert_decomposition_partition(checked, case, matrix, allocator, subjects)


def test_record_decomposition_replays_the_same_matrix_semantics() -> None:
    checked, case, matrix, _ = _matrix(
        "record Pair\n"
        "  left: bool\n"
        "  right: bool\n"
        "let value = Pair(left = true, right = false)\n"
        "case value of | Pair(left = true) => 1 | Pair(left = false) => 0"
    )

    compiled = compile_match_site(normalize_case(case, checked))
    assert isinstance(compiled.root, DecisionDecompose)
    assert compiled.root.demanded_occurrences == (compiled.root.children[0].id,)
    validate_compiled_case(compiled)
    assert matrix.rows


def test_record_decompositions_partition_partial_and_nested_patterns() -> None:
    checked, case, matrix, allocator = _matrix(
        "record Inner\n"
        "  value: decimal\n"
        "record Outer\n"
        "  inner: Inner\n"
        "  label: text\n"
        'let value = Outer(inner = Inner(value = 1), label = "x")\n'
        "case value of\n"
        "  | Outer(inner = Inner(value = 1)) => 1\n"
        '  | Outer(label = "x") => 2\n'
        "  | _ => 3\n"
    )
    outer = head_constructors(matrix, 0)[0]
    assert isinstance(outer, NominalConstructor)
    outer_nominal = NominalId(outer.record_type.decl_id)
    outer_result = specialize(matrix, 0, outer, allocator)
    inner = head_constructors(outer_result.matrix, 0)[0]
    assert isinstance(inner, NominalConstructor)
    inner_nominal = NominalId(inner.record_type.decl_id)

    subjects = (
        RecordValue(
            outer_nominal,
            outer.record_type.name,
            {
                "inner": RecordValue(inner_nominal, inner.record_type.name, {"value": IntValue(1)}),
                "label": TextValue("x"),
            },
        ),
        RecordValue(
            outer_nominal,
            outer.record_type.name,
            {
                "inner": RecordValue(
                    inner_nominal,
                    inner.record_type.name,
                    {"value": DecimalValue(decimal.Decimal("2"))},
                ),
                "label": TextValue("x"),
            },
        ),
        RecordValue(
            outer_nominal,
            outer.record_type.name,
            {
                "inner": RecordValue(
                    inner_nominal,
                    inner.record_type.name,
                    {"value": DecimalValue(decimal.Decimal("2"))},
                ),
                "label": TextValue("other"),
            },
        ),
    )
    inner_result = specialize(outer_result.matrix, 0, inner, outer_result.allocator)
    for subject in subjects:
        expected = reference_action(case, checked, subject)
        assert (
            matrix_action(
                inner_result.matrix,
                (subject.fields["inner"].fields["value"], subject.fields["label"]),
            )
            == expected
        )


def test_nested_enum_and_literal_decomposition_preserves_first_match_actions() -> None:
    checked, case, matrix, allocator = _matrix(
        "enum Payload\n"
        "  | number(value: decimal)\n"
        "  | word(value: text)\n"
        "enum Envelope\n"
        "  | wrapped(payload: Payload)\n"
        "  | empty\n"
        "let value: Envelope = wrapped(payload = number(value = 1))\n"
        "case value of\n"
        "  | wrapped(payload = number(value = 1)) => 1\n"
        "  | wrapped(payload = number(value = 2.5)) => 2\n"
        '  | wrapped(payload = word(value = "x")) => 3\n'
        "  | empty() => 4\n"
        "  | _ => 5\n"
    )
    envelope_heads = _constructor_by_variant(matrix, 0)
    wrapped = envelope_heads["wrapped"]
    empty = envelope_heads["empty"]
    envelope_type = cast(EnumType, matrix.occurrences[0].type)
    envelope_nominal = NominalId(envelope_type.decl_id)
    wrapped_cell = cast(ConstructorCell, matrix.rows[0].cells[0])
    payload_type = cast(EnumType, wrapped_cell.constructor.fields[0].type)
    payload_nominal = NominalId(payload_type.decl_id)

    def payload(variant: str, value: Value) -> RecordValue:
        return RecordValue(
            nominal=payload_nominal,
            display_name=f"{payload_type.name}::{variant}",
            fields={"value": value},
        )

    subjects = (
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{wrapped.terminal_name}",
            fields={"payload": payload("number", IntValue(1))},
        ),
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{wrapped.terminal_name}",
            fields={"payload": payload("number", DecimalValue(decimal.Decimal("1.0")))},
        ),
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{wrapped.terminal_name}",
            fields={"payload": payload("number", DecimalValue(decimal.Decimal("2.5")))},
        ),
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{wrapped.terminal_name}",
            fields={"payload": payload("word", TextValue("x"))},
        ),
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{wrapped.terminal_name}",
            fields={"payload": payload("word", TextValue("other"))},
        ),
        RecordValue(
            nominal=envelope_nominal,
            display_name=f"{envelope_type.name}::{empty.terminal_name}",
            fields={},
        ),
    )
    _assert_decomposition_partition(checked, case, matrix, allocator, subjects)
