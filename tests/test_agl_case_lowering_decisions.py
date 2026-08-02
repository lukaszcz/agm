"""Structural guarantees for decision-DAG to one-level IR lowering."""

from __future__ import annotations

import decimal
from dataclasses import replace

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir import (
    ExecutableProgram,
    IrBind,
    IrCase,
    IrConstUnit,
    IrField,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrSequence,
)
from agm.agl.matchcompile import DecisionDecompose
from agm.agl.matchcompile.model import (
    DecisionBranch,
    DecisionFail,
    DecisionSwitch,
    Occurrence,
    OccurrenceId,
    PathDecomposition,
)
from agm.agl.syntax.nodes import (
    ConstructorPattern,
    LiteralPattern,
    VarPattern,
    WildcardPattern,
)
from tests.agl.ir_harness import compile_checked_module, lower_compiled_module
from tests.agl.match_reference import case_sites
from tests.agl.module_graph import resolve_and_check_entry


def _lower(source: str) -> ExecutableProgram:
    capabilities = HostCapabilities(
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
            ),
        }
    )
    checked = resolve_and_check_entry(source, capabilities)
    compiled = compile_checked_module(checked)
    return lower_compiled_module(
        compiled,
        source_text=source,
        source_label="<test>",
    )


def _public_binding(program: ExecutableProgram, name: str) -> IrBind:
    for initializer in program.modules[program.entry_module].initializers:
        if isinstance(initializer, IrBind):
            if program.symbols[initializer.symbol].public_name == name:
                return initializer
            continue
        if not isinstance(initializer, IrSequence):
            continue
        root_capture, leaf = initializer.items
        if not isinstance(root_capture, IrBind) or not isinstance(leaf, IrSequence):
            continue
        binder = leaf.items[0]
        if isinstance(binder, IrBind) and program.symbols[binder.symbol].public_name == name:
            return root_capture
    raise AssertionError(f"missing public binding {name!r}")


def test_record_root_decomposition_projects_only_demanded_fields_without_a_switch() -> None:
    program = _lower(
        "record Box\n"
        "  first: int\n"
        "  second: int\n"
        "let value = Box(first = 1, second = 2)\n"
        "let result = case value of | Box(first = _ as selected) => selected\n"
        "()\n"
    )
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    decomposition = lowered.items[1]
    assert isinstance(decomposition, IrSequence)
    projection = decomposition.items[0]
    assert isinstance(projection, IrBind)
    assert isinstance(projection.value, IrField)
    assert projection.value.field == "first"
    assert not isinstance(decomposition, IrCase)


def test_lowering_rejects_a_forged_failed_decision(self_validation_disabled: None) -> None:
    """A failed decision path cannot become executable IR."""
    source = "let value = 1\ncase value of | _ => 1\n"
    checked = resolve_and_check_entry(source, HostCapabilities())
    compiled = compile_checked_module(checked)
    case_id, compiled_case = next(iter(case_sites(compiled.sites).items()))
    forged = replace(
        compiled,
        sites={**compiled.sites, case_id: replace(compiled_case, root=DecisionFail())},
    )
    with pytest.raises(AssertionError):
        lower_compiled_module(forged, source_text=source, source_label="<test>")


def test_lowering_rejects_a_forged_record_switch(self_validation_disabled: None) -> None:
    """Records are decomposition-only and cannot acquire a runtime case key."""
    source = (
        "record Box\n  value: int\nlet value = Box(value = 1)\n"
        "case value of | Box(value = _) => 1\n"
    )
    checked = resolve_and_check_entry(source, HostCapabilities())
    compiled = compile_checked_module(checked)
    case_id, compiled_case = next(iter(case_sites(compiled.sites).items()))
    root = compiled_case.root
    assert isinstance(root, DecisionDecompose)
    forged = replace(
        compiled,
        sites={
            **compiled.sites,
            case_id: replace(
                compiled_case,
                root=DecisionSwitch(
                    root.occurrence,
                    (DecisionBranch(root.constructor, root.child),),
                    None,
                ),
            ),
        },
    )
    with pytest.raises(AssertionError):
        lower_compiled_module(forged, source_text=source, source_label="<test>")


def test_wildcard_case_still_binds_root_subject_before_leaf() -> None:
    program = _lower("let value = 1\nlet result = case value of | _ => 2\n()")
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    root_binding, leaf = lowered.items
    assert isinstance(root_binding, IrBind)
    assert not program.symbols[root_binding.symbol].mutable
    assert program.symbols[root_binding.symbol].public_name is None
    assert not isinstance(leaf, IrCase)


def test_as_patterns_lower_every_chained_binder_at_the_matched_occurrence() -> None:
    program = _lower(
        "def select(value: int) -> int = case value of | 0 as captured => captured | _ => 0\n"
        "let value = 0\n"
        "let result = case value of | 0 as first as second => first + second | _ => select(value)\n"
        "()"
    )
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    switch = lowered.items[1]
    assert isinstance(switch, IrCase)
    body = switch.arms[0].body
    assert isinstance(body, IrSequence)
    bindings = [item for item in body.items if isinstance(item, IrBind)]
    assert len(bindings) == 2
    assert all(isinstance(binding.value, IrLoad) for binding in bindings)
    assert bindings[0].value.symbol == bindings[1].value.symbol


def test_singleton_enum_decomposition_projects_demanded_fields_before_nested_switch() -> None:
    program = _lower(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = true)\n"
        "let result = case value of\n"
        "  | pair(left = false) => 1\n"
        "  | pair(left = true) => 2\n"
        "()\n"
    )
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    decomposition = lowered.items[1]
    assert isinstance(decomposition, IrSequence)
    projection = decomposition.items[0]
    assert isinstance(projection, IrBind)
    assert isinstance(projection.value, IrField)
    assert projection.value.field == "left"
    nested_switch = decomposition.items[1]
    assert isinstance(nested_switch, IrCase)
    assert all(isinstance(arm.key, IrLiteralCaseKey) for arm in nested_switch.arms)


@pytest.mark.parametrize(
    ("source", "expected_kind", "expected_scalar"),
    [
        (
            "let value = true\nlet result = case value of | true => 1 | false => 0\n()",
            IrLiteralKind.BOOL,
            True,
        ),
        (
            "let value = 1\nlet result = case value of | 1 => 1 | _ => 0\n()",
            IrLiteralKind.NUMERIC,
            decimal.Decimal(1),
        ),
        (
            'let value = "x"\nlet result = case value of | "x" => 1 | _ => 0\n()',
            IrLiteralKind.TEXT,
            "x",
        ),
        (
            "let value: json = null\nlet result = case value of | null => 1 | _ => 0\n()",
            IrLiteralKind.NULL,
            None,
        ),
    ],
)
def test_literal_patterns_lower_to_canonical_one_level_keys(
    source: str,
    expected_kind: IrLiteralKind,
    expected_scalar: decimal.Decimal | bool | str | None,
) -> None:
    lowered = _public_binding(_lower(source), "result").value
    assert isinstance(lowered, IrSequence)
    switch = lowered.items[1]
    assert isinstance(switch, IrCase)
    keys = tuple(arm.key for arm in switch.arms)
    assert all(isinstance(key, IrLiteralCaseKey) for key in keys)
    assert IrLiteralCaseKey(expected_kind, expected_scalar) in keys


def test_shared_decision_node_remains_shared_ir_object() -> None:
    program = _lower(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = false)\n"
        "let result = case value of\n"
        "  | pair(left = false, right = false) => 1\n"
        "  | _ => 2\n"
        "()\n"
    )
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    decomposition = lowered.items[1]
    assert isinstance(decomposition, IrSequence)
    left_switch = decomposition.items[-1]
    assert isinstance(left_switch, IrCase)
    right_switch = left_switch.arms[0].body
    assert isinstance(right_switch, IrCase)
    assert right_switch.default is left_switch.default


def test_pattern_let_uses_the_shared_decision_dag_for_nested_demanded_fields() -> None:
    program = _lower(
        "record Leaf\n"
        "  value: int\n"
        "record Box\n"
        "  selected: Leaf\n"
        "  skipped: int\n"
        "let Box(selected = Leaf(value = value)) = Box(selected = Leaf(value = 7), skipped = 9)\n"
        "()\n"
    )
    (lowered, _) = program.modules[program.entry_module].initializers
    assert isinstance(lowered, IrSequence)
    root_capture, outer_decomposition = lowered.items
    assert isinstance(root_capture, IrBind)
    assert program.symbols[root_capture.symbol].public_name is None
    assert isinstance(outer_decomposition, IrSequence)
    outer_projection = outer_decomposition.items[0]
    assert isinstance(outer_projection, IrBind)
    assert isinstance(outer_projection.value, IrField)
    assert outer_projection.value.field == "selected"
    nested_decomposition = outer_decomposition.items[1]
    assert isinstance(nested_decomposition, IrSequence)
    nested_projection = nested_decomposition.items[0]
    assert isinstance(nested_projection, IrBind)
    assert isinstance(nested_projection.value, IrField)
    assert nested_projection.value.field == "value"
    leaf = nested_decomposition.items[1]
    assert isinstance(leaf, IrSequence)
    assert isinstance(leaf.items[-1], IrConstUnit)
    assert "skipped" not in [
        item.value.field
        for item in (*outer_decomposition.items, *nested_decomposition.items)
        if isinstance(item, IrBind) and isinstance(item.value, IrField)
    ]


def test_executable_ir_contains_no_source_pattern_or_occurrence_objects() -> None:
    program = _lower(
        "enum Box | box(value: int)\n"
        "let boxed = box(value = 2)\n"
        "let result = case boxed of | box(value = 1) => 1 | box(value = _ as n) => n\n"
        "()"
    )
    forbidden_names = tuple(
        item.__name__
        for item in (
            ConstructorPattern,
            LiteralPattern,
            VarPattern,
            WildcardPattern,
            Occurrence,
            OccurrenceId,
            PathDecomposition,
        )
    )
    executable_repr = repr(
        tuple(
            initializer
            for module in program.modules.values()
            for initializer in module.initializers
        )
    )
    assert all(f"{name}(" not in executable_repr for name in forbidden_names)
