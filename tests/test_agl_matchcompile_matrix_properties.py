"""Paper-derived and semantic partition properties for pattern matrices."""

from __future__ import annotations

import decimal
from dataclasses import replace
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.matchcompile import (
    BinderAssignment,
    Constructor,
    ConstructorCell,
    EnumConstructor,
    MatrixRow,
    OccurrenceAllocator,
    PatternMatrix,
    WildcardCell,
    default_matrix,
    head_constructors,
    matrix_from_normalized,
    normalize_case,
    specialize,
)
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.semantics.values import (
    BoolValue,
    DecimalValue,
    EnumValue,
    IntValue,
    JsonValue,
    TextValue,
    Value,
)
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedProgram, check
from tests.agl.match_reference import (
    canonical_cell_matches,
    matrix_action,
    reference_action,
)

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def _matrix(
    source: str,
) -> tuple[CheckedProgram, Case, PatternMatrix, OccurrenceAllocator]:
    checked = check(resolve(parse_program(source)), _CAPS)
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


def _constructor_by_variant(matrix: PatternMatrix, column: int) -> dict[str, EnumConstructor]:
    constructors = head_constructors(matrix, column)
    assert all(isinstance(constructor, EnumConstructor) for constructor in constructors)
    return {
        constructor.variant: constructor
        for constructor in cast(tuple[EnumConstructor, ...], constructors)
    }


def _migrated(row: MatrixRow, matrix: PatternMatrix, column: int) -> tuple[BinderAssignment, ...]:
    wildcard = cast(WildcardCell, row.cells[column])
    assert wildcard.binder is not None
    return (
        *row.binder_assignments,
        BinderAssignment(matrix.occurrences[column].id, wildcard.binder),
    )


def test_paper_specializations_preserve_complete_rows_and_priority() -> None:
    """Adapt Maranget's P, S(::, P), and S([], P) matrices to AgL enums."""
    _, _, root, allocator = _matrix(
        "enum List\n"
        "  | nil\n"
        "  | cons(head: int, tail: List)\n"
        "enum Subject\n"
        "  | subject(left: List, right: List)\n"
        "let value = subject(left = nil(), right = nil())\n"
        "case value of\n"
        "  | subject(left = nil(), right = _) => 1\n"
        "  | subject(left = left_value, right = nil()) => 2\n"
        "  | subject(left = cons(), right = cons()) => 3\n"
    )
    subject = cast(EnumConstructor, head_constructors(root, 0)[0])
    columns_result = specialize(root, 0, subject, allocator)
    columns = columns_result.matrix
    nil = _constructor_by_variant(columns, 0)["nil"]
    cons = _constructor_by_variant(columns, 0)["cons"]
    nil_result = specialize(columns, 0, nil, columns_result.allocator)
    cons_result = specialize(columns, 0, cons, nil_result.allocator)
    row_nil, row_wildcard, row_cons = columns.rows
    wildcard = cast(WildcardCell, row_wildcard.cells[0])
    explicit_cons = cast(ConstructorCell, row_cons.cells[0])

    expected_nil = (
        replace(row_nil, cells=(row_nil.cells[1],)),
        replace(
            row_wildcard,
            cells=(row_wildcard.cells[1],),
            binder_assignments=_migrated(row_wildcard, columns, 0),
        ),
    )
    expected_cons = (
        replace(
            row_wildcard,
            cells=(
                WildcardCell(None, wildcard.provenance),
                WildcardCell(None, wildcard.provenance),
                row_wildcard.cells[1],
            ),
            binder_assignments=_migrated(row_wildcard, columns, 0),
        ),
        replace(
            row_cons,
            cells=(*explicit_cons.arguments, row_cons.cells[1]),
        ),
    )
    expected_default = (
        replace(
            row_wildcard,
            cells=(row_wildcard.cells[1],),
            binder_assignments=_migrated(row_wildcard, columns, 0),
        ),
    )

    decompositions = (
        ("S(nil, P)", nil_result.matrix.rows, expected_nil),
        ("S(cons, P)", cons_result.matrix.rows, expected_cons),
        ("D(P)", default_matrix(columns, 0).rows, expected_default),
    )
    for name, actual, expected in decompositions:
        assert actual == expected, name


def test_paper_default_retains_and_migrates_all_wildcard_rows() -> None:
    """Adapt Maranget's Q and D(Q), retaining both wildcard-leading rows."""
    _, _, root, allocator = _matrix(
        "enum List\n"
        "  | nil\n"
        "  | cons(head: int, tail: List)\n"
        "enum Subject\n"
        "  | subject(left: List, right: List)\n"
        "let value = subject(left = nil(), right = nil())\n"
        "case value of\n"
        "  | subject(left = nil(), right = _) => 1\n"
        "  | subject(left = second_left, right = nil()) => 2\n"
        "  | subject(left = third_left, right = _) => 3\n"
    )
    subject = cast(EnumConstructor, head_constructors(root, 0)[0])
    matrix = specialize(root, 0, subject, allocator).matrix
    defaulted = default_matrix(matrix, 0)
    _, second, third = matrix.rows

    assert defaulted.rows == (
        replace(
            second,
            cells=(second.cells[1],),
            binder_assignments=_migrated(second, matrix, 0),
        ),
        replace(
            third,
            cells=(third.cells[1],),
            binder_assignments=_migrated(third, matrix, 0),
        ),
    )


def _head_arguments(
    constructor: Constructor,
    provenance_cell: ConstructorCell,
    value: Value,
) -> tuple[Value, ...] | None:
    head_only = ConstructorCell(
        constructor,
        tuple(WildcardCell(None, provenance_cell.provenance) for _ in range(constructor.arity)),
        provenance_cell.provenance,
    )
    if not canonical_cell_matches(head_only, value):
        return None
    if isinstance(constructor, EnumConstructor):
        assert isinstance(value, EnumValue)
        return tuple(value.fields[field.name] for field in constructor.fields)
    return ()


def _assert_decomposition_partition(
    checked: CheckedProgram,
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

    for subject in subjects:
        expected = reference_action(case, checked, subject)
        matching = [
            (specialized_matrix, arguments)
            for head, specialized_matrix, provenance_cell in specialized
            if (arguments := _head_arguments(head, provenance_cell, subject)) is not None
        ]
        if matching:
            assert len(matching) == 1
            specialized_matrix, arguments = matching[0]
            assert matrix_action(specialized_matrix, arguments) == expected
        else:
            assert matrix_action(defaulted, ()) == expected


def test_boolean_and_enum_decompositions_partition_complete_finite_domains() -> None:
    bool_checked, bool_case, bool_matrix, bool_allocator = _matrix(
        "let value = false\ncase value of | false => 1 | remaining => 2"
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
        "let value = red()\n"
        "case value of | red() => 1 | blue() => 2 | remaining => 3"
    )
    enum_type = cast(EnumConstructor, head_constructors(enum_matrix, 0)[0]).enum_type
    nominal = NominalId(enum_type.module_id, enum_type.name)
    subjects = tuple(
        EnumValue(nominal, enum_type.name, variant, {}) for variant in ("red", "green", "blue")
    )
    _assert_decomposition_partition(
        enum_checked, enum_case, enum_matrix, enum_allocator, subjects
    )


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


def test_nested_enum_and_literal_decomposition_preserves_first_match_actions() -> None:
    checked, case, matrix, allocator = _matrix(
        "enum Payload\n"
        "  | number(value: decimal)\n"
        "  | word(value: text)\n"
        "enum Envelope\n"
        "  | wrapped(payload: Payload)\n"
        "  | empty\n"
        "let value = wrapped(payload = number(value = 1))\n"
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
    envelope_nominal = NominalId(wrapped.enum_type.module_id, wrapped.enum_type.name)
    wrapped_cell = cast(ConstructorCell, matrix.rows[0].cells[0])
    payload_cell = cast(ConstructorCell, wrapped_cell.arguments[0])
    payload_type = cast(EnumConstructor, payload_cell.constructor).enum_type
    payload_nominal = NominalId(payload_type.module_id, payload_type.name)

    def payload(variant: str, value: Value) -> EnumValue:
        return EnumValue(payload_nominal, payload_type.name, variant, {"value": value})

    subjects = (
        EnumValue(
            envelope_nominal,
            wrapped.enum_type.name,
            wrapped.variant,
            {"payload": payload("number", IntValue(1))},
        ),
        EnumValue(
            envelope_nominal,
            wrapped.enum_type.name,
            wrapped.variant,
            {"payload": payload("number", DecimalValue(decimal.Decimal("1.0")))},
        ),
        EnumValue(
            envelope_nominal,
            wrapped.enum_type.name,
            wrapped.variant,
            {"payload": payload("number", DecimalValue(decimal.Decimal("2.5")))},
        ),
        EnumValue(
            envelope_nominal,
            wrapped.enum_type.name,
            wrapped.variant,
            {"payload": payload("word", TextValue("x"))},
        ),
        EnumValue(
            envelope_nominal,
            wrapped.enum_type.name,
            wrapped.variant,
            {"payload": payload("word", TextValue("other"))},
        ),
        EnumValue(envelope_nominal, empty.enum_type.name, empty.variant, {}),
    )
    _assert_decomposition_partition(checked, case, matrix, allocator, subjects)
