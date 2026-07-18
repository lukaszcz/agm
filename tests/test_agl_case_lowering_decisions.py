"""Structural guarantees for decision-DAG to one-level IR lowering."""

from __future__ import annotations

import decimal

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir import (
    ExecutableProgram,
    IrBind,
    IrCase,
    IrEnumCaseKey,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrSequence,
)
from agm.agl.lower import lower_module
from agm.agl.matchcompile import (
    MatchCompiledModule,
    compile_module_matches,
)
from agm.agl.matchcompile.model import Occurrence, OccurrenceId, PathDecomposition
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.syntax.nodes import (
    ConstructorPattern,
    LiteralPattern,
    VarPattern,
    WildcardPattern,
)
from agm.agl.typecheck import check_module


def _lower(source: str) -> ExecutableProgram:
    capabilities = HostCapabilities(
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
        }
    )
    checked = check_module(resolve_module(parse_program(source)), capabilities)
    result = compile_module_matches(checked)
    assert isinstance(result.compiled, MatchCompiledModule)
    return lower_module(
        result.compiled,
        source_text=source,
        source_label="<test>",
    )


def _public_binding(program: ExecutableProgram, name: str) -> IrBind:
    for initializer in program.modules[program.entry_module].initializers:
        if (
            isinstance(initializer, IrBind)
            and program.symbols[initializer.symbol].public_name == name
        ):
            return initializer
    raise AssertionError(f"missing public binding {name!r}")


def test_wildcard_case_still_binds_root_subject_before_leaf() -> None:
    program = _lower("let value = 1\nlet result = case value of | _ => 2\n()")
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    root_binding, leaf = lowered.items
    assert isinstance(root_binding, IrBind)
    assert not program.symbols[root_binding.symbol].mutable
    assert program.symbols[root_binding.symbol].public_name is None
    assert not isinstance(leaf, IrCase)


def test_enum_arm_binds_only_demanded_immediate_field_for_nested_switch() -> None:
    program = _lower(
        "enum Pair\n"
        "  | pair(left: bool, right: bool)\n"
        "let value = pair(left = false, right = true)\n"
        "let result = case value of\n"
        "  | pair(left = false) => 1\n"
        "  | _ => 2\n"
        "()\n"
    )
    lowered = _public_binding(program, "result").value
    assert isinstance(lowered, IrSequence)
    enum_switch = lowered.items[1]
    assert isinstance(enum_switch, IrCase)
    enum_arm = enum_switch.arms[0]
    assert isinstance(enum_arm.key, IrEnumCaseKey)
    assert tuple(name for name, _ in enum_arm.field_bindings) == ("left",)
    assert isinstance(enum_arm.body, IrCase)
    assert all(isinstance(arm.key, IrLiteralCaseKey) for arm in enum_arm.body.arms)


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
    enum_switch = lowered.items[1]
    assert isinstance(enum_switch, IrCase)
    left_switch = enum_switch.arms[0].body
    assert isinstance(left_switch, IrCase)
    right_switch = left_switch.arms[0].body
    assert isinstance(right_switch, IrCase)
    assert right_switch.default is left_switch.default


def test_executable_ir_contains_no_source_pattern_or_occurrence_objects() -> None:
    program = _lower(
        "enum Box | box(value: int)\n"
        "let boxed = box(value = 2)\n"
        "let result = case boxed of | box(value = 1) => 1 | box(value = n) => n\n"
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
