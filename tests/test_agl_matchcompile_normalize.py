"""Checked-pattern normalization contracts for the AgL match compiler."""

from __future__ import annotations

import decimal
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.matchcompile.model import (
    BinderAssignment,
    BoolConstructor,
    ClosedSignature,
    Constructor,
    ConstructorCell,
    DecisionBranch,
    DecisionFail,
    DecisionLeaf,
    DecisionSwitch,
    EnumConstructor,
    FieldOccurrenceProvenance,
    LiteralConstructor,
    LiteralKind,
    Occurrence,
    OccurrenceId,
    OmittedFieldProvenance,
    OpenSignature,
    SourcePatternProvenance,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import (
    MatchCompileInvariantError,
    constructor_inhabits_type,
    normalize_case,
    normalize_pattern,
    pattern_cell_inhabits_type,
    signature_for_type,
)
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.type_table import TypeDef
from agm.agl.semantics.types import (
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.agl.semantics.values import DecimalValue, EnumValue, TextValue
from agm.agl.syntax.nodes import Case, ConstructorPattern, Pattern
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedModule, check_module, check_program
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.match_reference import reference_action

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def _check(source: str) -> CheckedModule:
    return check_module(resolve_module(parse_program(source)), _CAPS)


def _only_case(program: object) -> Case:
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(program, collect)
    assert len(cases) == 1
    return cases[0]


def test_signatures_are_closed_for_boolean_and_enum_in_declaration_order() -> None:
    checked = _check(
        "enum Result[T]\n"
        "  | Empty\n"
        "  | Value(item: T, note: text)\n"
        "let value: Result[int] = Empty\n"
        "case value of | _ => 0"
    )
    table = checked.type_env.type_table

    bool_signature = signature_for_type(BoolType(), table)
    assert bool_signature == ClosedSignature(
        constructors=(BoolConstructor(False), BoolConstructor(True))
    )

    case = _only_case(checked.resolved.program)
    enum_type = checked.node_types[case.subject.node_id]
    assert isinstance(enum_type, EnumType)
    enum_signature = signature_for_type(enum_type, table)
    assert isinstance(enum_signature, ClosedSignature)
    assert [constructor.variant for constructor in enum_signature.constructors] == [
        "Empty",
        "Value",
    ]
    value = enum_signature.constructors[1]
    assert isinstance(value, EnumConstructor)
    assert value.enum_type == EnumType("Result", (IntType(),), ENTRY_ID)
    assert [(field.name, field.type) for field in value.fields] == [
        ("item", IntType()),
        ("note", TextType()),
    ]


@pytest.mark.parametrize(
    "subject_type",
    [
        TextType(),
        JsonType(),
        IntType(),
        DecimalType(),
        TypeVarType("T"),
        ListType(IntType()),
        DictType(IntType()),
        RecordType("R"),
        ExceptionType("E"),
        UnitType(),
        AgentType(),
        FunctionType((IntType(),), IntType()),
    ],
)
def test_every_current_non_closed_type_has_an_explicit_open_signature(
    subject_type: Type,
) -> None:
    checked = _check("()")
    assert signature_for_type(subject_type, checked.type_env.type_table) == OpenSignature()


def test_bottom_has_an_empty_closed_signature_and_no_inhabiting_patterns() -> None:
    checked = _check("()")

    assert signature_for_type(BottomType(), checked.type_env.type_table) == ClosedSignature(())
    assert not pattern_cell_inhabits_type(
        WildcardCell(
            binder=None,
            provenance=SourcePatternProvenance(0, SourceSpan(1, 1, 1, 1, 0, 0)),
        ),
        BottomType(),
    )


def test_flexible_inference_types_cannot_enter_match_normalization() -> None:
    checked = _check("()")
    leaked = InferenceVarType("T")

    with pytest.raises(MatchCompileInvariantError):
        signature_for_type(leaked, checked.type_env.type_table)
    with pytest.raises(MatchCompileInvariantError):
        constructor_inhabits_type(BoolConstructor(False), leaked)


def test_normalize_case_preserves_priority_actions_and_binder_provenance() -> None:
    checked = _check("let value = 1\ncase value of | 0 => 10 | captured => captured")
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    assert normalized.case_node_id == case.node_id
    assert normalized.type_table is checked.type_env.type_table
    assert normalized.occurrences == (normalized.root,)
    assert normalized.root.id.value == 0
    assert normalized.root.creation_order == 0
    assert [row.source_index for row in normalized.rows] == [0, 1]
    assert [row.action_id for row in normalized.rows] == [
        case.branches[0].node_id,
        case.branches[1].node_id,
    ]
    assert [action.body_node_id for action in normalized.actions] == [
        case.branches[0].body.node_id,
        case.branches[1].body.node_id,
    ]
    assert isinstance(normalized.rows[0].cells[0], ConstructorCell)
    binder_cell = normalized.rows[1].cells[0]
    assert isinstance(binder_cell, WildcardCell)
    assert binder_cell.binder is not None
    assert binder_cell.binder.node_id == case.branches[1].pattern.node_id
    assert binder_cell.binder.name == "captured"
    assert isinstance(binder_cell.provenance, SourcePatternProvenance)
    assert normalized.rows[1].binder_assignments == ()


def test_numeric_literals_share_runtime_equality_canonical_form() -> None:
    checked = _check("let value: decimal = 1\ncase value of | 1 => 10 | 1.0 => 20 | _ => 30")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)

    first = normalized.rows[0].cells[0]
    second = normalized.rows[1].cells[0]
    assert isinstance(first, ConstructorCell)
    assert isinstance(second, ConstructorCell)
    assert first.constructor == second.constructor
    assert first.constructor == LiteralConstructor(
        kind=LiteralKind.NUMERIC, value=decimal.Decimal("1")
    )


def test_fractional_decimal_arm_is_omitted_for_int_but_integral_decimal_is_retained() -> None:
    checked = _check("let value: int = 1\ncase value of | 1.5 => 15 | 1.0 => 10 | _ => 0")
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    assert [action.source_index for action in normalized.actions] == [0, 1, 2]
    assert [row.source_index for row in normalized.rows] == [1, 2]
    assert [row.action_id for row in normalized.rows] == [
        case.branches[1].node_id,
        case.branches[2].node_id,
    ]
    integral = normalized.rows[0].cells[0]
    assert isinstance(integral, ConstructorCell)
    assert integral.constructor == LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal("1.0"))


def test_nested_uninhabited_constructor_omits_only_its_source_row() -> None:
    checked = _check(
        "enum Box\n"
        "  | box(value: int)\n"
        "let subject: Box = box(value = 1)\n"
        "case subject of\n"
        "  | box(value = 1.5) => 15\n"
        "  | box(value = 1.0) => 10\n"
        "  | _ => 0\n"
    )
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    assert [action.source_index for action in normalized.actions] == [0, 1, 2]
    assert [row.source_index for row in normalized.rows] == [1, 2]
    assert [row.action_id for row in normalized.rows] == [
        case.branches[1].node_id,
        case.branches[2].node_id,
    ]


def test_constructor_inhabitation_is_total_over_current_constructor_and_type_unions() -> None:
    checked = _check("enum Choice\n  | none\nlet value: Choice = none\ncase value of | none => 0")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)
    enum_cell = normalized.rows[0].cells[0]
    assert isinstance(enum_cell, ConstructorCell)
    enum_constructor = enum_cell.constructor
    assert isinstance(enum_constructor, EnumConstructor)
    all_types: tuple[Type, ...] = (
        TextType(),
        JsonType(),
        BoolType(),
        IntType(),
        DecimalType(),
        TypeVarType("T"),
        ListType(IntType()),
        DictType(IntType()),
        RecordType("R"),
        EnumType("Choice"),
        ExceptionType("E"),
        UnitType(),
        AgentType(),
        FunctionType((IntType(),), IntType()),
        BottomType(),
    )
    constructors = (
        BoolConstructor(False),
        LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal("1")),
        LiteralConstructor(LiteralKind.TEXT, "x"),
        LiteralConstructor(LiteralKind.NULL, None),
        enum_constructor,
    )

    results = {
        (constructor, subject_type): constructor_inhabits_type(constructor, subject_type)
        for constructor in constructors
        for subject_type in all_types
    }

    assert results[BoolConstructor(False), BoolType()]
    assert results[constructors[1], IntType()]
    assert results[constructors[1], DecimalType()]
    assert results[constructors[2], TextType()]
    assert results[constructors[3], JsonType()]
    assert results[enum_constructor, EnumType("Choice")]
    assert sum(results.values()) == 6


@pytest.mark.parametrize(
    ("value", "inhabits_int", "inhabits_decimal"),
    [
        (decimal.Decimal("1"), True, True),
        (decimal.Decimal("1.0"), True, True),
        (decimal.Decimal("-2.000"), True, True),
        (decimal.Decimal("1.5"), False, True),
        (decimal.Decimal("NaN"), False, False),
        (decimal.Decimal("Infinity"), False, False),
        (decimal.Decimal("-Infinity"), False, False),
    ],
)
def test_numeric_constructor_inhabitation_matches_runtime_numeric_domains(
    value: decimal.Decimal,
    inhabits_int: bool,
    inhabits_decimal: bool,
) -> None:
    constructor = LiteralConstructor(LiteralKind.NUMERIC, value)

    assert constructor_inhabits_type(constructor, IntType()) is inhabits_int
    assert constructor_inhabits_type(constructor, DecimalType()) is inhabits_decimal


def test_boolean_literals_normalize_to_boolean_constructors() -> None:
    checked = _check("let value = true\ncase value of | false => 0 | true => 1")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)

    constructors = []
    for row in normalized.rows:
        cell = row.cells[0]
        assert isinstance(cell, ConstructorCell)
        constructors.append(cell.constructor)
    assert constructors == [BoolConstructor(False), BoolConstructor(True)]


def test_constructor_normalization_expands_omitted_generic_fields_in_declaration_order() -> None:
    checked = _check(
        "enum Item[T]\n"
        "  | Made(first: T, second: text, third: int)\n"
        'let value: Item[int] = Made(first = 1, second = "x", third = 3)\n'
        "case value of | Made(first = captured, third = 3) => captured | _ => 0"
    )
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    outer = normalized.rows[0].cells[0]
    assert isinstance(outer, ConstructorCell)
    assert isinstance(outer.constructor, EnumConstructor)
    assert [field.name for field in outer.constructor.fields] == [
        "first",
        "second",
        "third",
    ]
    assert len(outer.arguments) == 3
    first, second, third = outer.arguments
    assert isinstance(first, WildcardCell)
    assert first.binder is not None
    assert first.binder.name == "captured"
    assert isinstance(second, WildcardCell)
    assert second.binder is None
    assert second.provenance == OmittedFieldProvenance(
        constructor_pattern_id=case.branches[0].pattern.node_id,
        field_name="second",
        span=case.branches[0].pattern.span,
    )
    assert isinstance(third, ConstructorCell)
    assert third.constructor == LiteralConstructor(
        kind=LiteralKind.NUMERIC, value=decimal.Decimal("3")
    )


def test_bare_nullary_variant_uses_resolver_classification() -> None:
    checked = _check(
        "enum Choice\n  | none\n  | some(value: int)\n"
        "let value: Choice = none\n"
        "case value of | none => 0 | some(_) => 1"
    )
    case = _only_case(checked.resolved.program)
    normalized = normalize_case(case, checked)

    first = normalized.rows[0].cells[0]
    assert isinstance(first, ConstructorCell)
    assert isinstance(first.constructor, EnumConstructor)
    assert first.constructor.variant == "none"
    assert first.arguments == ()
    assert isinstance(normalized.rows[1].cells[0], ConstructorCell)


def test_imported_generic_enum_normalizes_from_checked_metadata(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "lib": "enum Choice[T]\n  | absent\n  | present(value: T, note: text)",
            "entry": (
                "import lib\n"
                'let value: Choice[int] = present(value = 1, note = "x")\n'
                "case value of\n"
                "  | present(value = captured) => captured\n"
                "  | absent => 0\n"
            ),
        },
    )
    checked = check_program(resolve_program(graph), _CAPS)
    checked = checked.modules[ENTRY_ID]
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    constructor_cell = normalized.rows[0].cells[0]
    assert isinstance(constructor_cell, ConstructorCell)
    constructor = constructor_cell.constructor
    assert isinstance(constructor, EnumConstructor)
    assert constructor.enum_type == EnumType("Choice", (IntType(),), ModuleId.from_dotted("lib"))
    assert [(field.name, field.type) for field in constructor.fields] == [
        ("value", IntType()),
        ("note", TextType()),
    ]
    assert len(constructor_cell.arguments) == 2
    omitted = constructor_cell.arguments[1]
    assert isinstance(omitted, WildcardCell)
    assert isinstance(omitted.provenance, OmittedFieldProvenance)


def test_text_and_null_literals_retain_distinct_typed_canonical_keys() -> None:
    text_checked = _check('let value = "x"\ncase value of | "x" => 1 | _ => 0')
    text_cell = (
        normalize_case(_only_case(text_checked.resolved.program), text_checked).rows[0].cells[0]
    )
    assert isinstance(text_cell, ConstructorCell)
    assert text_cell.constructor == LiteralConstructor(LiteralKind.TEXT, "x")

    null_checked = _check("let value: json = null\ncase value of | null => 1 | _ => 0")
    null_cell = (
        normalize_case(_only_case(null_checked.resolved.program), null_checked).rows[0].cells[0]
    )
    assert isinstance(null_cell, ConstructorCell)
    assert null_cell.constructor == LiteralConstructor(LiteralKind.NULL, None)


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (LiteralKind.NUMERIC, "1"),
        (LiteralKind.TEXT, None),
        (LiteralKind.NULL, "null"),
    ],
)
def test_literal_constructor_rejects_noncanonical_payloads(
    kind: LiteralKind, value: object
) -> None:
    with pytest.raises(ValueError, match="invalid value"):
        LiteralConstructor(kind, cast("decimal.Decimal | str | None", value))


def test_model_rejects_invalid_occurrences_cells_and_normalized_matrices() -> None:
    checked = _check("let value = 1\ncase value of | 1 => 1 | _ => 0")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)
    source_cell = normalized.rows[0].cells[0]
    assert isinstance(source_cell, ConstructorCell)

    with pytest.raises(ValueError, match="occurrence ids"):
        OccurrenceId(-1)
    with pytest.raises(ValueError, match="creation order"):
        replace(normalized.root, creation_order=-1)
    with pytest.raises(ValueError, match="argument count"):
        replace(source_cell, arguments=(source_cell,))
    with pytest.raises(ValueError, match="only its root"):
        replace(normalized, occurrences=())
    with pytest.raises(ValueError, match="row width"):
        bad_row = replace(normalized.rows[0], cells=())
        replace(normalized, rows=(bad_row, normalized.rows[1]))
    with pytest.raises(ValueError, match="rows must retain"):
        bad_row = replace(normalized.rows[0], source_index=1)
        replace(normalized, rows=(bad_row, normalized.rows[1]))
    with pytest.raises(ValueError, match="actions must retain"):
        bad_action = replace(normalized.actions[0], source_index=1)
        replace(normalized, actions=(bad_action, normalized.actions[1]))
    with pytest.raises(ValueError, match="rows and source actions"):
        bad_action = replace(normalized.actions[0], action_id=-1)
        replace(normalized, actions=(bad_action, normalized.actions[1]))

    # Rows are the surviving ordered, unique subsequence of source actions.
    assert replace(normalized, rows=(normalized.rows[1],)).rows[0].source_index == 1
    with pytest.raises(ValueError, match="ordered unique subsequence"):
        replace(normalized, rows=(normalized.rows[1], normalized.rows[0]))
    with pytest.raises(ValueError, match="ordered unique subsequence"):
        replace(normalized, rows=(normalized.rows[0], normalized.rows[0]))
    with pytest.raises(ValueError, match="source action"):
        replace(normalized, rows=(replace(normalized.rows[0], action_id=-1),))


def test_decision_model_carries_occurrence_and_binder_identities() -> None:
    checked = _check("let value = 1\ncase value of | captured => captured")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)
    cell = normalized.rows[0].cells[0]
    assert isinstance(cell, WildcardCell)
    assert cell.binder is not None
    assignment = BinderAssignment(normalized.root.id, cell.binder)
    leaf = DecisionLeaf(normalized.rows[0].action_id, (assignment,))
    fail = DecisionFail()
    constructor = LiteralConstructor(LiteralKind.NUMERIC, decimal.Decimal(1))
    switch = DecisionSwitch(
        normalized.root,
        (DecisionBranch(constructor, leaf),),
        fail,
    )
    child = Occurrence(
        id=OccurrenceId(1),
        creation_order=1,
        type=IntType(),
        provenance=FieldOccurrenceProvenance(
            parent=normalized.root.id,
            constructor=constructor,
            field_name="value",
            field_index=0,
            source=cell.provenance,
        ),
    )

    assert switch.keyed_children[0].decision is leaf
    assert switch.default is fail
    assert leaf.binder_assignments == (assignment,)
    assert child.provenance.parent == normalized.root.id


@dataclass(frozen=True)
class _UnknownPattern:
    node_id: int
    span: SourceSpan


def _replace_case_pattern(case: Case, pattern: Pattern) -> Case:
    branch = replace(case.branches[0], pattern=pattern)
    return replace(case, branches=(branch, *case.branches[1:]))


def test_signature_and_pattern_dispatch_reject_unknown_future_members() -> None:
    checked = _check("let value = 1\ncase value of | _ => 0")
    case = _only_case(checked.resolved.program)

    with pytest.raises(MatchCompileInvariantError, match="unsupported semantic type"):
        signature_for_type(cast(Type, object()), checked.type_env.type_table)
    with pytest.raises(MatchCompileInvariantError, match="unsupported semantic type"):
        constructor_inhabits_type(BoolConstructor(False), cast(Type, object()))
    with pytest.raises(MatchCompileInvariantError, match="unsupported constructor"):
        constructor_inhabits_type(cast(Constructor, object()), IntType())
    unknown = cast(Pattern, _UnknownPattern(node_id=999, span=case.span))
    with pytest.raises(MatchCompileInvariantError, match="unsupported source pattern"):
        normalize_pattern(unknown, IntType(), checked)


def test_missing_enum_and_subject_metadata_raise_compiler_invariants() -> None:
    checked = _check("let value = 1\ncase value of | _ => 0")
    case = _only_case(checked.resolved.program)

    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        signature_for_type(EnumType("Missing"), checked.type_env.type_table)
    without_subject = replace(
        checked,
        node_types={
            node_id: node_type
            for node_id, node_type in checked.node_types.items()
            if node_id != case.subject.node_id
        },
    )
    with pytest.raises(MatchCompileInvariantError, match="missing checked subject type"):
        normalize_case(case, without_subject)


def test_malformed_checked_literal_and_bare_variant_metadata_raise_invariants() -> None:
    literal_checked = _check("let value = 1\ncase value of | 1 => 1 | _ => 0")
    literal_case = _only_case(literal_checked.resolved.program)
    wrong_literal_type = replace(
        literal_checked,
        node_types={**literal_checked.node_types, literal_case.subject.node_id: BoolType()},
    )
    with pytest.raises(MatchCompileInvariantError, match="incompatible"):
        normalize_case(literal_case, wrong_literal_type)

    enum_checked = _check(
        "enum Choice\n  | none\n  | some(value: int)\n"
        "let value: Choice = none\ncase value of | none => 0 | _ => 1"
    )
    enum_case = _only_case(enum_checked.resolved.program)
    non_enum = replace(
        enum_checked,
        node_types={**enum_checked.node_types, enum_case.subject.node_id: IntType()},
    )
    with pytest.raises(MatchCompileInvariantError, match="non-enum"):
        normalize_case(enum_case, non_enum)

    bare_none = enum_case.branches[0].pattern
    bare_some = replace(bare_none, name="some")
    assert bare_none.node_id in enum_checked.resolved.bare_variant_refs
    some_ref = replace(enum_checked.resolved.bare_variant_refs[bare_none.node_id], variant="some")
    malformed_resolved = replace(
        enum_checked.resolved,
        bare_variant_refs={bare_none.node_id: some_ref},
    )
    malformed_checked = replace(enum_checked, resolved=malformed_resolved)
    with pytest.raises(MatchCompileInvariantError, match="not nullary"):
        normalize_case(_replace_case_pattern(enum_case, bare_some), malformed_checked)


def test_bare_variant_normalization_rejects_missing_and_wrong_owner_metadata() -> None:
    checked = _check(
        "enum Choice\n  | none\nlet value: Choice = none\ncase value of | none => 0 | _ => 1"
    )
    case = _only_case(checked.resolved.program)
    pattern = case.branches[0].pattern

    missing_ref = replace(checked.resolved, bare_variant_refs={})
    with pytest.raises(MatchCompileInvariantError, match="missing or inconsistent"):
        normalize_case(case, replace(checked, resolved=missing_ref))

    ref = checked.resolved.bare_variant_refs[pattern.node_id]
    checked.type_env.type_table.register(
        TypeDef(kind="enum", name="Other", module_id=ENTRY_ID, variants=(("none", ()),))
    )
    wrong_owner = replace(ref, owner_name="Other")
    wrong_owner_resolved = replace(
        checked.resolved, bare_variant_refs={pattern.node_id: wrong_owner}
    )
    with pytest.raises(MatchCompileInvariantError, match="owner does not match"):
        normalize_case(case, replace(checked, resolved=wrong_owner_resolved))


def test_malformed_checked_constructor_metadata_raise_invariants() -> None:
    checked = _check(
        "enum Choice\n  | some(value: int)\n"
        "let value: Choice = some(value = 1)\n"
        "case value of | some(value = captured) => captured | _ => 0"
    )
    case = _only_case(checked.resolved.program)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    supplied_pairs = checked.argument_bindings.constructor_patterns[pattern.node_id]

    non_enum = replace(
        checked,
        node_types={**checked.node_types, case.subject.node_id: IntType()},
    )
    with pytest.raises(MatchCompileInvariantError, match="non-enum"):
        normalize_case(case, non_enum)

    missing_type = replace(
        checked,
        node_types={**checked.node_types, case.subject.node_id: EnumType("Missing")},
    )
    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        normalize_case(case, missing_type)

    unknown_variant = replace(pattern, name="missing")
    with pytest.raises(MatchCompileInvariantError, match="unknown variant"):
        normalize_case(_replace_case_pattern(case, unknown_variant), checked)

    no_bindings = replace(
        checked.argument_bindings,
        constructor_patterns={
            node_id: pairs
            for node_id, pairs in checked.argument_bindings.constructor_patterns.items()
            if node_id != pattern.node_id
        },
    )
    with pytest.raises(MatchCompileInvariantError, match="missing checked argument"):
        normalize_case(case, replace(checked, argument_bindings=no_bindings))

    field_name, child_pattern = supplied_pairs[0]
    duplicate = replace(
        checked.argument_bindings,
        constructor_patterns={
            **checked.argument_bindings.constructor_patterns,
            pattern.node_id: ((field_name, child_pattern), (field_name, child_pattern)),
        },
    )
    with pytest.raises(MatchCompileInvariantError, match="duplicate checked field"):
        normalize_case(case, replace(checked, argument_bindings=duplicate))

    unknown = replace(
        checked.argument_bindings,
        constructor_patterns={
            **checked.argument_bindings.constructor_patterns,
            pattern.node_id: (("missing", child_pattern),),
        },
    )
    with pytest.raises(MatchCompileInvariantError, match="unknown fields"):
        normalize_case(case, replace(checked, argument_bindings=unknown))


def test_source_reference_matcher_preserves_priority_and_partial_constructor_fields() -> None:
    checked = _check(
        "enum Choice\n  | absent\n  | present(value: decimal, note: text)\n"
        'let value: Choice = present(value = 1.0, note = "x")\n'
        "case value of\n"
        "  | present(value = 1) => 10\n"
        "  | present(note = captured) => 20\n"
        "  | absent => 30\n"
        "  | _ => 40\n"
    )
    case = _only_case(checked.resolved.program)
    enum_type = checked.node_types[case.subject.node_id]
    assert isinstance(enum_type, EnumType)
    nominal = NominalId(enum_type.module_id, enum_type.name)

    assert (
        reference_action(
            case,
            checked,
            EnumValue(
                nominal,
                "Choice",
                "present",
                {"value": DecimalValue(decimal.Decimal("1.0")), "note": TextValue("x")},
            ),
        )
        == case.branches[0].node_id
    )
    assert (
        reference_action(
            case,
            checked,
            EnumValue(
                nominal,
                "Choice",
                "present",
                {"value": DecimalValue(decimal.Decimal("2")), "note": TextValue("x")},
            ),
        )
        == case.branches[1].node_id
    )
    assert (
        reference_action(
            case,
            checked,
            EnumValue(nominal, "Choice", "absent", {}),
        )
        == case.branches[2].node_id
    )
