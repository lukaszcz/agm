"""Checked-pattern normalization contracts for the AgL match compiler."""

from __future__ import annotations

import decimal
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import NO_DECL_ID
from agm.agl.matchcompile.model import (
    BinderAssignment,
    BoolConstructor,
    CaseSite,
    ClosedSignature,
    Constructor,
    ConstructorCell,
    ConstructorField,
    DecisionBranch,
    DecisionFail,
    DecisionLeaf,
    DecisionSwitch,
    FieldOccurrenceProvenance,
    LetSite,
    LiteralConstructor,
    LiteralKind,
    NominalConstructor,
    Occurrence,
    OccurrenceId,
    OmittedFieldProvenance,
    SourcePatternProvenance,
    WildcardCell,
)
from agm.agl.matchcompile.normalize import (
    MatchCompileInvariantError,
    constructor_inhabits_type,
    enum_constructor,
    normalize_case,
    normalize_let,
    normalize_pattern,
    pattern_cell_inhabits_type,
    signature_for_type,
)
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import (
    BoolType,
    BottomType,
    DecimalType,
    EnumType,
    InferenceVarType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
)
from agm.agl.semantics.values import DecimalValue, RecordValue, TextValue
from agm.agl.syntax.nodes import AsPattern, Case, ConstructorPattern, LetDecl, Pattern
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedModule, check_program
from tests._agl_helpers import enum_typedef, next_decl_id, register_typedef, strip_decl_ids
from tests.agl.ir_harness import make_graph_from_files
from tests.agl.match_reference import reference_action
from tests.agl.module_graph import resolve_and_check_inline_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def _check(source: str) -> CheckedModule:
    return resolve_and_check_inline_entry(source, _CAPS)


def _only_case(program: object) -> Case:
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(program, collect)
    assert len(cases) == 1
    return cases[0]


def _stripped_signature(signature: ClosedSignature) -> ClosedSignature:
    """Return *signature* with every embedded nominal handle's ``decl_id`` reset.

    ``strip_decl_ids`` only walks a single ``Type``'s structural children, so
    it cannot reach the ``RecordType``/``EnumType`` handles nested inside a
    ``NominalConstructor``/``NominalConstructor`` (and its ``ConstructorField``s)
    directly; this reapplies it field-by-field across the closed signature
    the match compiler returns.
    """

    def strip(constructor: Constructor) -> Constructor:
        if isinstance(constructor, NominalConstructor):
            return replace(
                constructor,
                record_type=cast(RecordType, strip_decl_ids(constructor.record_type)),
                fields=tuple(
                    replace(field, type=strip_decl_ids(field.type)) for field in constructor.fields
                ),
            )
        return constructor

    return replace(signature, constructors=tuple(strip(c) for c in signature.constructors))


def test_let_normalization_retains_its_initializer_pattern_and_matched_type() -> None:
    checked = _check("let value: bool = true")
    lets: list[LetDecl] = []

    def collect(node: object) -> None:
        if isinstance(node, LetDecl):
            lets.append(node)

    walk(checked.resolved.program, collect)
    (let,) = lets

    normalized = normalize_let(let, checked)

    assert normalized.site_node_id == let.node_id
    assert isinstance(normalized.source, LetSite)
    assert normalized.root.type == checked.let_matched_types[let.node_id]
    assert normalized.source.action.action_id == normalized.rows[0].action_id
    assert normalized.source.actions == (normalized.source.action,)
    assert len(normalized.rows) == len(normalized.source.actions) == 1
    assert normalized.root.provenance.subject_node_id == let.value.node_id
    assert normalized.rows[0].source_pattern_id == let.pattern.node_id
    assert normalized.root.provenance.site_node_id == let.node_id
    with pytest.raises(MatchCompileInvariantError, match="matched type"):
        normalize_let(let, replace(checked, let_matched_types={}))


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
    assert [constructor.terminal_name for constructor in enum_signature.constructors] == [
        "Empty",
        "Value",
    ]
    value = enum_signature.constructors[1]
    assert isinstance(value, NominalConstructor)
    assert strip_decl_ids(value.record_type) == strip_decl_ids(table.enum_members(enum_type)[1])
    assert [(field.name, field.type) for field in value.fields] == [
        ("item", IntType()),
        ("note", TextType()),
    ]


def test_record_signature_and_pattern_normalization_use_canonical_nominal_identity() -> None:
    checked = _check(
        "record Inner[T]\n"
        "  value: T\n"
        "record Outer[T]\n"
        "  first: T\n"
        "  inner: Inner[T]\n"
        "  label: text\n"
        "type Alias[T] = Outer[T]\n"
        'let value: Alias[int] = Outer(first = 1, inner = Inner(value = 2), label = "x")\n'
        "case value of | Alias(inner = Inner(value = _ as captured)) as whole => captured"
    )
    case = _only_case(checked.resolved.program)
    subject_type = checked.node_types[case.subject.node_id]
    assert isinstance(subject_type, RecordType)

    signature = signature_for_type(subject_type, checked.type_env.type_table)
    assert isinstance(signature, ClosedSignature)

    assert _stripped_signature(signature) == ClosedSignature(
        (
            NominalConstructor(
                RecordType("Outer", (IntType(),), ENTRY_ID),
                (
                    ConstructorField("first", IntType()),
                    ConstructorField("inner", RecordType("Inner", (IntType(),), ENTRY_ID)),
                    ConstructorField("label", TextType()),
                ),
            ),
        )
    )
    outer = normalize_case(case, checked).rows[0].cells[0]
    assert isinstance(outer, ConstructorCell)
    assert isinstance(outer.constructor, NominalConstructor)
    assert strip_decl_ids(outer.constructor.record_type) == RecordType(
        "Outer", (IntType(),), ENTRY_ID
    )
    assert [field.name for field in outer.constructor.fields] == ["first", "inner", "label"]
    assert isinstance(outer.arguments[0], WildcardCell)
    inner = outer.arguments[1]
    assert isinstance(inner, ConstructorCell)
    assert isinstance(inner.constructor, NominalConstructor)
    assert [binder.name for binder in inner.arguments[0].binders] == ["captured"]
    assert isinstance(outer.arguments[2], WildcardCell)
    assert [binder.name for binder in outer.binders] == ["whole"]

    source_pattern = case.branches[0].pattern
    assert isinstance(source_pattern, AsPattern)
    constructor_pattern = source_pattern.pattern
    assert isinstance(constructor_pattern, ConstructorPattern)
    constructor_ref = checked.pattern_constructor_ref_for(constructor_pattern.node_id)
    assert constructor_ref is not None
    assert checked.pattern_constructor_owner_for(constructor_pattern.node_id) == NominalId(
        subject_type.decl_id
    )
    with pytest.raises(MatchCompileInvariantError, match="missing final constructor"):
        normalize_case(
            case,
            replace(checked, pattern_constructor_refs={}),
        )
    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(
            case,
            replace(
                checked,
                pattern_constructor_owners={
                    constructor_pattern.node_id: NominalId(inner.constructor.record_type.decl_id)
                },
            ),
        )


def test_record_signatures_keep_modules_and_generic_instantiations_distinct() -> None:
    table = TypeTable()
    left_module = ModuleId.from_path("left")
    right_module = ModuleId.from_path("right")
    left_def = TypeDef(
        kind="record",
        name="Box",
        module_id=left_module,
        type_params=("T",),
        fields=(("value", TypeVarType("T")),),
        decl_node_id=next_decl_id(),
    )
    right_def = TypeDef(
        kind="record",
        name="Box",
        module_id=right_module,
        type_params=("T",),
        fields=(("value", TypeVarType("T")),),
        decl_node_id=next_decl_id(),
    )
    table.register(left_def)
    table.register(right_def)

    left_int = signature_for_type(left_def.handle((IntType(),)), table)
    left_text = signature_for_type(left_def.handle((TextType(),)), table)
    right_int = signature_for_type(right_def.handle((IntType(),)), table)
    assert isinstance(left_int, ClosedSignature)

    assert left_int != left_text
    assert left_int != right_int
    assert _stripped_signature(left_int) == ClosedSignature(
        (
            NominalConstructor(
                RecordType("Box", (IntType(),), left_module),
                (ConstructorField("value", IntType()),),
            ),
        )
    )


def test_record_signature_cache_preserves_identity_and_invalidates_redeclarations() -> None:
    """Internal robustness guard: a declaration's shape is immutable once
    registered in production (a redeclaration always mints a fresh identity,
    never rewrites an existing one -- see ``TypeTable.register``), so the only
    still-live way to change a def under an existing identity is
    ``merge_from`` treating another table as authoritative. This proves the
    signature cache is keyed by declaration VALUE, not identity alone, using
    that path.
    """
    table = TypeTable()
    box_id = next_decl_id()
    first_def = TypeDef(
        kind="record",
        name="Box",
        module_id=ENTRY_ID,
        fields=(("value", IntType()),),
        decl_node_id=box_id,
    )
    table.register(first_def)
    handle = first_def.handle()

    original = signature_for_type(handle, table)

    assert original is signature_for_type(handle, table)

    source = TypeTable()
    source.register(replace(first_def, fields=(("label", TextType()),)))
    table.merge_from(source)

    redeclared = signature_for_type(handle, table)
    assert isinstance(redeclared, ClosedSignature)

    assert redeclared is not original
    assert _stripped_signature(redeclared) == ClosedSignature(
        (NominalConstructor(RecordType("Box"), (ConstructorField("label", TextType()),)),)
    )


def test_scoped_record_signature_cache_invalidates_on_its_own_redeclaration() -> None:
    """A cached signature is keyed by the declaration its handle names, not by
    whatever root declaration happens to share the bare name, so redeclaring
    the scoped type alone still invalidates it.

    Internal robustness guard, same as the root-declaration case above: uses
    ``merge_from`` to change a def under an existing identity, the only path
    still live for that in production.
    """
    table = TypeTable()
    table.register(
        TypeDef(
            kind="record",
            name="Box",
            module_id=ENTRY_ID,
            fields=(("root", IntType()),),
            decl_node_id=next_decl_id(),
        )
    )
    scoped_id = next_decl_id()
    scoped_def = TypeDef(
        kind="record",
        name="Box",
        module_id=ENTRY_ID,
        scope_path=("A",),
        fields=(("value", IntType()),),
        decl_node_id=scoped_id,
    )
    table.register(scoped_def)
    handle = scoped_def.handle()

    original = signature_for_type(handle, table)
    assert original is signature_for_type(handle, table)

    source = TypeTable()
    source.register(replace(scoped_def, fields=(("label", TextType()),)))
    table.merge_from(source)

    redeclared = signature_for_type(handle, table)
    assert isinstance(redeclared, ClosedSignature)
    assert redeclared is not original
    assert _stripped_signature(redeclared) == ClosedSignature(
        (
            NominalConstructor(
                RecordType("Box", scope_path=("A",)),
                (ConstructorField("label", TextType()),),
            ),
        )
    )


def test_bottom_has_an_empty_closed_signature_and_no_inhabiting_patterns() -> None:
    checked = _check("()")

    assert signature_for_type(BottomType(), checked.type_env.type_table) == ClosedSignature(())
    assert not pattern_cell_inhabits_type(
        WildcardCell(
            provenance=SourcePatternProvenance(0, SourceSpan(1, 1, 1, 1, 0, 0)),
        ),
        BottomType(),
        checked.type_env.type_table,
    )


def test_flexible_inference_types_cannot_enter_match_normalization() -> None:
    checked = _check("()")
    leaked = InferenceVarType("T")

    with pytest.raises(MatchCompileInvariantError):
        signature_for_type(leaked, checked.type_env.type_table)
    with pytest.raises(MatchCompileInvariantError):
        constructor_inhabits_type(BoolConstructor(False), leaked, checked.type_env.type_table)


def test_normalize_case_preserves_priority_actions_and_binder_provenance() -> None:
    checked = _check("let value = 1\ncase value of | 0 => 10 | _ as captured => captured")
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    assert normalized.site_node_id == case.node_id
    assert normalized.type_table is checked.type_env.type_table
    assert normalized.occurrences == (normalized.root,)
    assert normalized.root.id.value == 0
    assert normalized.root.creation_order == 0
    assert [row.source_index for row in normalized.rows] == [0, 1]
    assert [row.action_id for row in normalized.rows] == [
        case.branches[0].node_id,
        case.branches[1].node_id,
    ]
    assert isinstance(normalized.source, CaseSite)
    assert [action.body_node_id for action in normalized.source.actions] == [
        case.branches[0].body.node_id,
        case.branches[1].body.node_id,
    ]
    assert isinstance(normalized.rows[0].cells[0], ConstructorCell)
    binder_cell = normalized.rows[1].cells[0]
    assert isinstance(binder_cell, WildcardCell)
    assert len(binder_cell.binders) == 1
    assert binder_cell.binders[0].node_id == case.branches[1].pattern.node_id
    assert binder_cell.binders[0].name == "captured"
    assert isinstance(binder_cell.provenance, SourcePatternProvenance)
    assert normalized.rows[1].binder_assignments == ()


def test_as_patterns_preserve_all_current_occurrence_binders() -> None:
    checked = _check(
        "enum E\n  | A(value: int)\n  | B\nlet value: E = A(1)\n"
        "case value of | A(value = _ as inner) as whole as same => inner | B => 0"
    )
    normalized = normalize_case(_only_case(checked.resolved.program), checked)

    cell = normalized.rows[0].cells[0]
    assert isinstance(cell, ConstructorCell)
    assert [binder.name for binder in cell.binders] == ["whole", "same"]
    child = cell.arguments[0]
    assert isinstance(child, WildcardCell)
    assert [binder.name for binder in child.binders] == ["inner"]


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

    assert isinstance(normalized.source, CaseSite)
    assert [action.source_index for action in normalized.source.actions] == [0, 1, 2]
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

    assert isinstance(normalized.source, CaseSite)
    assert [action.source_index for action in normalized.source.actions] == [0, 1, 2]
    assert [row.source_index for row in normalized.rows] == [1, 2]
    assert [row.action_id for row in normalized.rows] == [
        case.branches[1].node_id,
        case.branches[2].node_id,
    ]


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

    assert constructor_inhabits_type(constructor, IntType(), TypeTable()) is inhabits_int
    assert constructor_inhabits_type(constructor, DecimalType(), TypeTable()) is inhabits_decimal


def test_non_data_and_generic_types_have_no_inhabiting_constructors() -> None:
    """Match compilation cannot construct values for non-concrete subject types."""
    assert not constructor_inhabits_type(BoolConstructor(False), TypeVarType("T"), TypeTable())


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
        "case value of | Made(first = _ as captured, third = 3) => captured | _ => 0"
    )
    case = _only_case(checked.resolved.program)

    normalized = normalize_case(case, checked)

    outer = normalized.rows[0].cells[0]
    assert isinstance(outer, ConstructorCell)
    assert isinstance(outer.constructor, NominalConstructor)
    assert [field.name for field in outer.constructor.fields] == [
        "first",
        "second",
        "third",
    ]
    assert len(outer.arguments) == 3
    first, second, third = outer.arguments
    assert isinstance(first, WildcardCell)
    assert [binder.name for binder in first.binders] == ["captured"]
    assert isinstance(second, WildcardCell)
    assert second.binders == ()
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
    assert isinstance(first.constructor, NominalConstructor)
    assert first.constructor.terminal_name == "none"
    assert first.arguments == ()
    assert isinstance(normalized.rows[1].cells[0], ConstructorCell)


def test_imported_generic_enum_normalizes_from_checked_metadata(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "lib": "enum Choice[T]\n  | absent\n  | present(value: T, note: text)",
            "entry": (
                "import lib::*\n"
                'let value: Choice[int] = present(value = 1, note = "x")\n'
                "case value of\n"
                "  | present(value = _ as captured) => captured\n"
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
    assert isinstance(constructor, NominalConstructor)
    assert strip_decl_ids(constructor.record_type) == strip_decl_ids(
        checked.type_env.type_table.enum_members(
            cast(EnumType, checked.node_types[case.subject.node_id])
        )[1]
    )
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
        assert isinstance(normalized.source, CaseSite)
        bad_action = replace(normalized.source.actions[0], source_index=1)
        replace(
            normalized,
            source=replace(normalized.source, actions=(bad_action, normalized.source.actions[1])),
        )
    with pytest.raises(ValueError, match="rows and match-site actions"):
        assert isinstance(normalized.source, CaseSite)
        bad_action = replace(normalized.source.actions[0], action_id=-1)
        replace(
            normalized,
            source=replace(normalized.source, actions=(bad_action, normalized.source.actions[1])),
        )

    # Rows are the surviving ordered, unique subsequence of match-site actions.
    assert replace(normalized, rows=(normalized.rows[1],)).rows[0].source_index == 1
    with pytest.raises(ValueError, match="ordered unique subsequence"):
        replace(normalized, rows=(normalized.rows[1], normalized.rows[0]))
    with pytest.raises(ValueError, match="ordered unique subsequence"):
        replace(normalized, rows=(normalized.rows[0], normalized.rows[0]))
    with pytest.raises(ValueError, match="match-site action"):
        replace(normalized, rows=(replace(normalized.rows[0], action_id=-1),))


def test_decision_model_carries_occurrence_and_binder_identities() -> None:
    checked = _check("let value = 1\ncase value of | _ as captured => captured")
    normalized = normalize_case(_only_case(checked.resolved.program), checked)
    cell = normalized.rows[0].cells[0]
    assert isinstance(cell, WildcardCell)
    assert len(cell.binders) == 1
    assignment = BinderAssignment(normalized.root.id, cell.binders[0])
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
        constructor_inhabits_type(
            BoolConstructor(False), cast(Type, object()), checked.type_env.type_table
        )
    with pytest.raises(MatchCompileInvariantError, match="unsupported constructor"):
        constructor_inhabits_type(
            cast(Constructor, object()), IntType(), checked.type_env.type_table
        )
    unknown = cast(Pattern, _UnknownPattern(node_id=999, span=case.span))
    with pytest.raises(MatchCompileInvariantError, match="unsupported source pattern"):
        normalize_pattern(unknown, IntType(), checked)


def test_missing_enum_and_subject_metadata_raise_compiler_invariants() -> None:
    checked = _check(
        "enum E\n  | A(value: int)\nlet value: E = A(value = 1)\ncase value of | A(_) => 0"
    )
    case = _only_case(checked.resolved.program)

    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        enum_constructor(EnumType("Missing"), "missing", checked.type_env.type_table)
    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        signature_for_type(EnumType("Missing"), checked.type_env.type_table)
    checked.type_env.type_table.register(
        TypeDef(kind="record", name="E", module_id=ENTRY_ID, decl_node_id=999)
    )
    wrong_kind = replace(
        checked,
        node_types={
            **checked.node_types,
            case.subject.node_id: EnumType("E", decl_id=999),
        },
    )
    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        normalize_case(case, wrong_kind)
    with pytest.raises(MatchCompileInvariantError, match="cannot resolve record signature"):
        signature_for_type(RecordType("Missing"), checked.type_env.type_table)
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
    bare_none_ref = enum_checked.pattern_classifications[bare_none.node_id]
    assert bare_none_ref is not None
    some_ref = replace(bare_none_ref, owner_name="some", owner_decl_node_id=-1)
    malformed_checked = replace(enum_checked, pattern_classifications={bare_none.node_id: some_ref})
    with pytest.raises(MatchCompileInvariantError, match="invalid final"):
        normalize_case(_replace_case_pattern(enum_case, bare_some), malformed_checked)


def test_bare_variant_normalization_rejects_missing_and_wrong_owner_metadata() -> None:
    checked = _check(
        "enum Choice\n  | none\nlet value: Choice = none\ncase value of | none => 0 | _ => 1"
    )
    case = _only_case(checked.resolved.program)
    pattern = case.branches[0].pattern

    with pytest.raises(MatchCompileInvariantError, match="missing final constructor"):
        normalize_case(case, replace(checked, pattern_classifications={}))

    ref = checked.pattern_classifications[pattern.node_id]
    assert ref is not None
    register_typedef(
        checked.type_env.type_table, enum_typedef("Other", {"none": {}}, module_id=ENTRY_ID)
    )
    wrong_owner = replace(ref, owner_name="Other")
    with pytest.raises(MatchCompileInvariantError, match="invalid final"):
        normalize_case(
            case, replace(checked, pattern_classifications={pattern.node_id: wrong_owner})
        )


def test_malformed_checked_constructor_metadata_raise_invariants() -> None:
    checked = _check(
        "enum Choice\n  | some(value: int)\n  | other(value: int)\n"
        "let value: Choice = some(value = 1)\n"
        "case value of | some(value = _ as captured) => captured | _ => 0"
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

    missing_enum_type = EnumType("Missing")
    missing_type = replace(
        checked,
        node_types={**checked.node_types, case.subject.node_id: missing_enum_type},
        pattern_constructor_owners={pattern.node_id: NominalId(missing_enum_type.decl_id)},
    )
    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        normalize_case(case, missing_type)

    unknown_variant = replace(pattern, name="missing")
    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(_replace_case_pattern(case, unknown_variant), checked)
    choice_type = checked.node_types[case.subject.node_id]
    assert isinstance(choice_type, EnumType)
    with pytest.raises(MatchCompileInvariantError, match="unknown variant"):
        enum_constructor(choice_type, "missing", checked.type_env.type_table)

    different_variant = replace(pattern, name="other")
    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(_replace_case_pattern(case, different_variant), checked)

    choice_type = checked.node_types[case.subject.node_id]
    assert isinstance(choice_type, EnumType)
    constructor_ref = checked.pattern_constructor_ref_for(pattern.node_id)
    assert constructor_ref is not None
    member = checked.type_env.type_table.enum_member_names(choice_type)[pattern.name]
    assert checked.pattern_constructor_owner_for(pattern.node_id) == NominalId(member.decl_id)
    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(
            case,
            replace(
                checked,
                pattern_constructor_refs={
                    pattern.node_id: replace(constructor_ref, owner_name="not-the-pattern-member")
                },
            ),
        )
    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(
            case,
            replace(
                checked,
                pattern_constructor_owners={pattern.node_id: NominalId(NO_DECL_ID)},
            ),
        )

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


def test_renamed_constructor_rejects_unknown_canonical_variant_metadata() -> None:
    checked = _check(
        "use S::{E::some as X}\n"
        "\n"
        "scope S\n"
        "  enum E | some(value: int)\n"
        "end S\n"
        "\n"
        "let value: S::E = X(value = 1)\n"
        "case value of | X(value = _ as captured) => captured | _ => 0"
    )
    case = _only_case(checked.resolved.program)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    constructor_ref = checked.pattern_constructor_ref_for(pattern.node_id)
    assert constructor_ref is not None

    with pytest.raises(MatchCompileInvariantError, match="invalid final constructor"):
        normalize_case(
            case,
            replace(
                checked,
                pattern_constructor_refs={
                    pattern.node_id: replace(constructor_ref, owner_name="missing")
                },
            ),
        )


def test_source_reference_matcher_preserves_priority_and_partial_constructor_fields() -> None:
    checked = _check(
        "enum Choice\n  | absent\n  | present(value: decimal, note: text)\n"
        'let value: Choice = present(value = 1.0, note = "x")\n'
        "case value of\n"
        "  | present(value = 1) => 10\n"
        "  | present(note = _ as captured) => 20\n"
        "  | absent => 30\n"
        "  | _ => 40\n"
    )
    case = _only_case(checked.resolved.program)
    enum_type = checked.node_types[case.subject.node_id]
    assert isinstance(enum_type, EnumType)
    nominal = NominalId(checked.type_env.type_table.enum_member_names(enum_type)["present"].decl_id)

    assert (
        reference_action(
            case,
            checked,
            RecordValue(
                nominal=nominal,
                fields={"value": DecimalValue(decimal.Decimal("1.0")), "note": TextValue("x")},
            ),
        )
        == case.branches[0].node_id
    )
    assert (
        reference_action(
            case,
            checked,
            RecordValue(
                nominal=nominal,
                fields={"value": DecimalValue(decimal.Decimal("2")), "note": TextValue("x")},
            ),
        )
        == case.branches[1].node_id
    )
    assert (
        reference_action(
            case,
            checked,
            RecordValue(
                nominal=NominalId(
                    checked.type_env.type_table.enum_member_names(enum_type)["absent"].decl_id
                ),
                fields={},
            ),
        )
        == case.branches[2].node_id
    )
