"""Behavior contracts for field-directed bare pattern names."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.typecheck import AglTypeError, check_module
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import evaluate_ir, make_graph_from_files


def _check(source: str) -> None:
    check_module(resolve_module(parse_program(source)), HostCapabilities())


def _reject_scope(source: str) -> None:
    with pytest.raises(AglScopeError):
        resolve_module(parse_program(source))


def _reject_type(source: str) -> None:
    with pytest.raises(AglTypeError):
        _check(source)


def test_top_level_bare_patterns_are_constructor_only() -> None:
    _check("enum Flag\n  | on\nlet on = 1\nlet value: Flag = Flag::on\ncase value of | on => 1")
    _reject_scope("let value = 1\ncase value of | typo => 1")
    _reject_scope(
        "enum Flag\n  | on\nlet typo = 1\nlet value: Flag = Flag::on\ncase value of | typo => 1"
    )


def test_final_classification_restores_outer_value_references() -> None:
    result = evaluate_ir(
        "enum Flag\n  | on\n  | off\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let on = 7\nlet value = packet(Flag::on())\n"
        "let result = case value of | packet(on) => on | packet(_) => 0\nresult"
    )
    assert result["result"].value == 7


def test_lowering_uses_selected_binder_slot_for_assignment() -> None:
    result = evaluate_ir(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "var on = 1\nlet item = packet(Flag::on)\n"
        "let result = case item of | packet(on) =>\n"
        "  on := on + 1\n"
        "  on\n"
        "result"
    )

    assert result["result"].value == 2


def test_lowering_uses_selected_constructor_slot_for_calls() -> None:
    result = evaluate_ir(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let item = packet(Flag::on)\n"
        "let result = case item of | packet(on) => on() == Flag::on\n"
        "result"
    )

    assert result["result"].value is True


def test_nested_names_follow_the_matched_field() -> None:
    _check(
        "enum Flag\n  | on\n  | off\n"
        "enum Packet\n  | packet(first: int, second: int, flag: Flag, label: text)\n"
        'let value = packet(1, 2, on(), "ok")\n'
        "case value of | packet(first, second, flag = on, label = label as renamed) => renamed"
    )
    _reject_type(
        "enum Packet\n  | packet(first: int, second: int)\n"
        "let value = packet(1, 2)\ncase value of | packet(second, first) => first"
    )
    _reject_type(
        "enum Packet\n  | packet(first: int, second: int)\n"
        "let value = packet(1, 2)\ncase value of | packet(first = second) => second"
    )
    _reject_type(
        "enum Packet\n  | packet(first: int)\n"
        "let value = packet(1)\ncase value of | packet(first = typo) => typo"
    )


def test_nested_constructor_and_binder_clashes_are_field_directed() -> None:
    _check(
        "enum Flag\n  | on\n  | off\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let value = packet(on())\ncase value of | packet(on) => 1 | packet(_) => 0"
    )
    _check(
        "enum Other\n  | value\n"
        "enum Packet\n  | packet(value: int)\n"
        "let item = packet(1)\ncase item of | packet(value) => value"
    )
    _reject_type(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(on: Flag)\n"
        "let value = packet(on())\ncase value of | packet(on) => 1"
    )
    _check(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(on: Flag)\n"
        "let value = packet(on())\ncase value of | packet(on()) => 1"
    )
    _check(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let value = packet(on())\ncase value of | Packet::packet(flag = Flag::on) => 1"
    )
    _check(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(on: Flag)\n"
        "let value = packet(on())\ncase value of | packet(_ as on) => 1"
    )
    _check(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let value = packet(on())\ncase value of | packet(on) => on"
    )
    _check(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let value = packet(on())\ncase value of | packet(on as on) => on"
    )


def test_constructor_candidate_after_explicit_binder_is_field_directed() -> None:
    result = evaluate_ir(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(value: int, flag: Flag)\n"
        "let item = packet(7, on())\n"
        "let result = case item of | packet(_ as on, on) => on\n"
        "result"
    )
    assert result["result"].value == 7
    _reject_type(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(value: int, on: int)\n"
        "let item = packet(7, 8)\n"
        "case item of | packet(_ as on, on) => on"
    )


def test_scope_rejects_duplicate_binders_and_typecheck_rejects_ambiguous_outer_references() -> None:
    _reject_scope(
        "enum Packet\n  | packet(value: int)\n"
        "let item = packet(1)\ncase item of | packet(value as value) => value"
    )
    _reject_type(
        "enum Flag\n  | on\nenum Other\n  | on\n"
        "enum Packet\n  | packet(flag: Flag)\n"
        "let item = packet(Flag::on)\ncase item of | packet(on) => on"
    )


def test_cross_module_and_builtin_fields_use_field_directed_classification(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "library": "enum Flag\n  | on\n  | off",
            "entry": (
                "open import library\n"
                "enum Packet\n  | packet(flag: library::Flag)\n"
                "let on = 7\nlet item = packet(library::Flag::on)\n"
                "case item of | packet(on) => on"
            ),
        },
    )
    check_program(resolve_program(graph), HostCapabilities())
    _check(
        "enum Holder\n  | holder(policy: ParsePolicy)\n"
        "let item = holder(ParsePolicy::Retry(n = 1))\n"
        "case item of | holder(ParsePolicy::Retry(n)) => n"
    )


def test_named_only_pattern_shorthand_uses_the_same_field_rule() -> None:
    _check(
        "enum Packet\n  | packet(*, value: int)\n"
        "let item = packet(value = 1)\ncase item of | packet(value) => value"
    )
    _reject_type(
        "enum Flag\n  | on\n"
        "enum Packet\n  | packet(*, on: Flag)\n"
        "let item = packet(on = on())\ncase item of | packet(on) => 1"
    )
