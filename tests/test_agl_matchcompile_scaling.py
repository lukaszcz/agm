"""Deterministic complexity regressions for pattern-match compilation."""

from __future__ import annotations

import pytest

import agm.agl.matchcompile.compiler as compiler_module
import agm.agl.matchcompile.matrix as matrix_module
from agm.agl.capabilities import HostCapabilities
from agm.agl.matchcompile.compiler import compile_case
from agm.agl.matchcompile.matrix import (
    OccurrenceAllocator,
    head_constructors,
    matrix_from_normalized,
    specialize,
)
from agm.agl.matchcompile.normalize import normalize_case
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.semantics.types import BoolType, EnumType
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck import CheckedModule, check_module

_CAPS = HostCapabilities(
    agent_names=frozenset(),
    has_default_agent=True,
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
    },
)


def _normalized(source: str) -> tuple[CheckedModule, Case]:
    checked = check_module(resolve_module(parse_program(source)), _CAPS)
    cases: list[Case] = []

    def collect(node: object) -> None:
        if isinstance(node, Case):
            cases.append(node)

    walk(checked.resolved.program, collect)
    assert len(cases) == 1
    return checked, cases[0]


def test_specializing_many_heads_only_classifies_source_rows_once(
    monkeypatch: pytest.MonkeyPatch,
    self_validation_disabled: None,
) -> None:
    head_count = 80
    variants = "\n".join(f"  | v{index}" for index in range(head_count))
    branches = "\n".join(f"  | v{index}() => {index}" for index in range(head_count))
    checked, case = _normalized(
        f"enum Wide\n{variants}\nlet subject = v0()\ncase subject of\n{branches}"
    )
    normalized = normalize_case(case, checked)
    matrix = matrix_from_normalized(normalized)
    heads = head_constructors(matrix, 0)
    allocator = OccurrenceAllocator.for_case(normalized)
    classifications = 0
    original_hash = EnumType.__hash__

    def counted_hash(enum_type: EnumType) -> int:
        nonlocal classifications
        classifications += 1
        return original_hash(enum_type)

    monkeypatch.setattr(EnumType, "__hash__", counted_hash)
    action_ids: list[int] = []
    for head in heads:
        specialized = specialize(matrix, 0, head, allocator)
        allocator = specialized.allocator
        action_ids.append(specialized.matrix.rows[0].action_id)

    assert action_ids == [branch.node_id for branch in case.branches]
    assert classifications <= head_count * 8


def test_terminal_wide_constructor_state_skips_column_profiles(
    monkeypatch: pytest.MonkeyPatch,
    self_validation_disabled: None,
) -> None:
    field_count = 120
    fields = ", ".join(f"f{index}: bool" for index in range(field_count))
    arguments = ", ".join(f"f{index} = true" for index in range(field_count))
    checked, case = _normalized(
        f"enum Wide\n  | wide({fields})\n"
        f"let subject = wide({arguments})\n"
        "case subject of | wide() => 1"
    )
    normalized = normalize_case(case, checked)
    profile_builds = 0
    original_build = matrix_module._build_column_profiles

    def counted_build(
        matrix: matrix_module.PatternMatrix,
    ) -> tuple[matrix_module._ColumnProfile, ...]:
        nonlocal profile_builds
        profile_builds += 1
        return original_build(matrix)

    monkeypatch.setattr(matrix_module, "_build_column_profiles", counted_build)

    compiled = compile_case(normalized)

    assert compiled.reachable_action_ids == (case.branches[0].node_id,)
    assert profile_builds == 1


def test_compile_state_keys_cache_structural_hash_work(
    monkeypatch: pytest.MonkeyPatch,
    self_validation_disabled: None,
) -> None:
    checked, case = _normalized("case true of | true => 1 | false => 2")
    normalized = normalize_case(case, checked)
    hash_calls = 0
    original_hash = BoolType.__hash__

    def counted_hash(value: BoolType) -> int:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(value)

    monkeypatch.setattr(BoolType, "__hash__", counted_hash)
    first = matrix_from_normalized(normalized)
    second = matrix_from_normalized(normalized)
    first_key = compiler_module._compile_state_key(first)
    second_key = compiler_module._compile_state_key(second)

    for _ in range(256):
        assert hash(first_key) == hash(second_key)
    assert first_key == second_key
    assert hash_calls <= 2
