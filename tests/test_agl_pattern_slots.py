"""Atomic authority tests for field-directed pattern slots."""

from dataclasses import replace

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.symbols import AglScopeError, BinderKind, BindingRef, PatternSlot, SlotCandidate
from agm.agl.syntax.nodes import (
    AsPattern,
    AssignStmt,
    Block,
    Case,
    ConstructorPattern,
    LetDecl,
    VarPattern,
    VarRef,
)
from agm.agl.typecheck import AglTypeError, TypeEnvironment, check_module
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.checker import _Checker


def _resolve(source: str):
    return resolve_module(parse_program(source))


def test_pattern_slot_model_and_empty_scope_tables() -> None:
    resolved = _resolve("let value = 1\nvalue")
    binding = next(iter(resolved.resolution.values()))
    candidate = SlotCandidate(11, resolved.program.span, True)
    slot = PatternSlot(7, "value", (candidate,), binding, 2)

    assert slot.candidates == (candidate,)
    assert slot.alternative is binding
    assert BinderKind.pattern_slot.value == "pattern_slot"
    assert resolved.pattern_slots == {}
    assert not hasattr(resolved, "provisional_pattern_binders")
    assert not hasattr(resolved, "case_scopes")
    assert not hasattr(resolved, "slot_references")


def test_scope_binds_one_shared_pattern_slot_per_branch_name() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(left: Flag, right: Flag)\n"
        "let item = packet(on(), on())\n"
        "case item of\n"
        "  | packet(on, on) => on\n"
        "  | packet(on, on) => on"
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    first, second = case.branches
    assert isinstance(first.pattern, ConstructorPattern)
    assert isinstance(second.pattern, ConstructorPattern)
    first_bares = first.pattern.positional
    second_bares = second.pattern.positional
    assert all(isinstance(pattern, VarPattern) for pattern in (*first_bares, *second_bares))
    assert isinstance(first.body, VarRef)
    assert isinstance(second.body, VarRef)

    slots = tuple(resolved.pattern_slots.values())
    assert len(slots) == 2
    assert [slot.name for slot in slots] == ["on", "on"]
    assert [candidate.pattern_node_id for candidate in slots[0].candidates] == [
        pattern.node_id for pattern in first_bares
    ]
    assert [candidate.pattern_node_id for candidate in slots[1].candidates] == [
        pattern.node_id for pattern in second_bares
    ]
    assert resolved.resolution[first.body.node_id].kind is BinderKind.pattern_slot
    assert resolved.resolution[first.body.node_id].slot_id == slots[0].slot_id
    assert resolved.resolution[second.body.node_id].kind is BinderKind.pattern_slot
    assert resolved.resolution[second.body.node_id].slot_id == slots[1].slot_id


def test_scope_slot_covers_as_candidates_and_all_branch_body_references() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(value: Flag)\n"
        "let item = packet(on())\n"
        "case item of\n"
        "  | packet(on as on) =>\n"
        "    let first = on\n"
        "    on"
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    as_pattern = pattern.positional[0]
    assert isinstance(as_pattern, AsPattern)
    assert isinstance(as_pattern.pattern, VarPattern)
    body = case.branches[0].body
    assert isinstance(body, Block)
    first = body.items[0]
    assert isinstance(first, LetDecl)
    assert isinstance(first.value, VarRef)
    assert isinstance(body.items[1], VarRef)

    slot = next(iter(resolved.pattern_slots.values()))
    assert [candidate.pattern_node_id for candidate in slot.candidates] == [
        as_pattern.pattern.node_id,
        as_pattern.node_id,
    ]
    assert [candidate.unconditional for candidate in slot.candidates] == [False, True]
    for ref in (first.value, body.items[1]):
        assert resolved.resolution[ref.node_id].kind is BinderKind.pattern_slot
        assert resolved.resolution[ref.node_id].slot_id == slot.slot_id


def test_scope_slot_alternative_is_the_outer_slot_ref() -> None:
    resolved = _resolve(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(1)\n"
        "case item of\n"
        "  | packet(value) =>\n"
        "    case item of\n"
        "      | packet(value) => value"
    )
    outer_case = resolved.program.body.items[-1]
    assert isinstance(outer_case, Case)
    outer_body = outer_case.branches[0].body
    assert isinstance(outer_body, Block)
    inner_case = outer_body.items[0]
    assert isinstance(inner_case, Case)
    slots = tuple(resolved.pattern_slots.values())
    assert len(slots) == 2
    assert slots[1].alternative is not None
    assert slots[1].alternative.kind is BinderKind.pattern_slot
    assert slots[1].alternative.slot_id == slots[0].slot_id


@pytest.mark.parametrize(
    ("pattern", "candidate_facts"),
    (
        ("packet(value, _ as value)", (True, False)),
        ("packet(_ as value, value)", (False, True)),
    ),
)
def test_scope_defers_duplicate_binders_for_local_nullary_enum_variants(
    pattern: str, candidate_facts: tuple[bool, bool]
) -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | value\n"
        "enum Packet\n"
        "  | packet(value: int, other: int)\n"
        "let item = packet(1, 2)\n"
        f"case item of | {pattern} => value"
    )

    slot = next(iter(resolved.pattern_slots.values()))
    assert tuple(
        candidate.can_match_bare_pattern for candidate in slot.candidates
    ) == candidate_facts
    assert resolved.constructor_candidates["value"][0].can_match_bare_pattern
    with pytest.raises(AglTypeError):
        check_module(resolved, HostCapabilities())


@pytest.mark.parametrize("pattern", ("packet(value, _ as value)", "packet(_ as value, value)"))
def test_scope_rejects_duplicate_binders_for_local_payload_variant_regardless_of_order(
    pattern: str,
) -> None:
    with pytest.raises(AglScopeError):
        _resolve(
            "enum Flag\n"
            "  | value(data: int)\n"
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            f"case item of | {pattern} => value"
        )


@pytest.mark.parametrize("pattern", ("packet(Retry, _ as Retry)", "packet(_ as Retry, Retry)"))
def test_scope_rejects_duplicate_binders_for_prelude_payload_variant_regardless_of_order(
    pattern: str,
) -> None:
    with pytest.raises(AglScopeError):
        _resolve(
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            f"case item of | {pattern} => Retry"
        )


@pytest.mark.parametrize("pattern", ("packet(Token, _ as Token)", "packet(_ as Token, Token)"))
def test_scope_rejects_duplicate_binders_for_record_candidates_regardless_of_order(
    pattern: str,
) -> None:
    with pytest.raises(AglScopeError):
        _resolve(
            "record Token\n"
            "  value: int\n"
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            f"case item of | {pattern} => Token"
        )


@pytest.mark.parametrize(
    "pattern",
    ("packet(ExecResult, _ as ExecResult)", "packet(_ as ExecResult, ExecResult)"),
)
def test_scope_rejects_duplicate_binders_for_prelude_record_candidates_regardless_of_order(
    pattern: str,
) -> None:
    with pytest.raises(AglScopeError):
        _resolve(
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            f"case item of | {pattern} => ExecResult"
        )


def test_constructor_candidate_metadata_is_only_true_for_nullary_enum_variants() -> None:
    resolved = _resolve(
        "record Token\n"
        "  value: int\n"
        "enum Flag\n"
        "  | empty\n"
        "  | payload(value: int)\n"
        "()"
    )

    assert resolved.constructor_candidates["Token"][0].can_match_bare_pattern is False
    assert resolved.constructor_candidates["empty"][0].can_match_bare_pattern is True
    assert resolved.constructor_candidates["payload"][0].can_match_bare_pattern is False
    assert resolved.constructor_candidates["Retry"][0].can_match_bare_pattern is False
    assert resolved.constructor_candidates["ExecResult"][0].can_match_bare_pattern is False


def test_checked_accessors_fully_dereference_binder_slot_without_mutating_scope() -> None:
    resolved = _resolve(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(1)\n"
        "case item of | packet(value) => value"
    )
    before_resolution = dict(resolved.resolution)
    before_constructors = dict(resolved.constructor_refs)
    body_ref = next(
        node_id
        for node_id, ref in resolved.resolution.items()
        if ref.kind is BinderKind.pattern_slot
    )

    checked = check_module(resolved, HostCapabilities())
    raw = checked.resolved.resolution[body_ref]
    assert raw.kind is BinderKind.pattern_slot
    assert raw.slot_id is not None
    binding = checked.binding_for(body_ref)
    assert binding == checked.slot_resolution[raw.slot_id]
    assert binding is not None
    assert binding.kind is BinderKind.pattern_binding
    assert checked.constructor_ref_for(body_ref) is None
    assert resolved.resolution == before_resolution
    assert resolved.constructor_refs == before_constructors
    assert checked.resolved is resolved


def test_checked_accessors_fully_dereference_constructor_slot() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(on())\n"
        "case item of | packet(on) => on"
    )
    body_ref = next(
        node_id
        for node_id, ref in resolved.resolution.items()
        if ref.kind is BinderKind.pattern_slot
    )

    checked = check_module(resolved, HostCapabilities())
    raw = checked.resolved.resolution[body_ref]
    assert raw.kind is BinderKind.pattern_slot
    assert raw.slot_id is not None
    binding = checked.binding_for(body_ref)
    assert binding == checked.slot_resolution[raw.slot_id]
    assert binding is not None
    assert binding.kind is BinderKind.constructor_binding
    assert checked.constructor_ref_for(body_ref) == checked.slot_constructor_refs[raw.slot_id]


def test_constructor_call_uses_selected_pattern_slot() -> None:
    checked = check_module(
        _resolve(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on()"
        ),
        HostCapabilities(),
    )

    assert checked.resolved.program is not None


def test_checked_accessors_dereference_nested_slot_alternatives() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "var on: int = 0\n"
        "let item = packet(Flag::on)\n"
        "case item of\n"
        "  | packet(on) =>\n"
        "    case item of | packet(on) => on\n"
        "  | packet(_) => 0"
    )
    slot_refs = [
        node_id
        for node_id, ref in resolved.resolution.items()
        if ref.kind is BinderKind.pattern_slot
    ]
    checked = check_module(resolved, HostCapabilities())

    assert len(slot_refs) == 1
    binding = checked.binding_for(slot_refs[0])
    assert binding is not None
    assert binding.kind is BinderKind.var_binding
    assert checked.constructor_ref_for(slot_refs[0]) is None


def test_slot_selection_rejects_binderless_slot_without_alternative() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(on())\n"
        "case item of | packet(on) => on"
    )
    slot = next(iter(resolved.pattern_slots.values()))

    with pytest.raises(AssertionError):
        check_module(
            replace(resolved, pattern_slots={slot.slot_id: replace(slot, alternative=None)}),
            HostCapabilities(),
        )


def test_slot_selection_rejects_nested_slot_without_selected_final_alternative() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "var on: int = 0\n"
        "let item = packet(Flag::on)\n"
        "case item of\n"
        "  | packet(on) =>\n"
        "    case item of\n"
        "      | packet(on) => on\n"
        "  | packet(_) => 0"
    )
    outer_slot, inner_slot = resolved.pattern_slots.values()
    assert inner_slot.alternative is not None
    assert inner_slot.alternative.kind is BinderKind.pattern_slot
    malformed_alternative = replace(inner_slot.alternative, slot_id=inner_slot.slot_id + 1)
    malformed = replace(
        resolved,
        pattern_slots={
            outer_slot.slot_id: outer_slot,
            inner_slot.slot_id: replace(inner_slot, alternative=malformed_alternative),
        },
    )

    with pytest.raises(AssertionError):
        check_module(malformed, HostCapabilities())


def test_inference_rollback_discards_nested_slot_selections_without_mutating_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(Flag::on)\n"
        "case item of\n"
        "  | packet(on) =>\n"
        "    case item of\n"
        "      | packet(on) => on\n"
        "      | packet(_) => Flag::on\n"
        "  | packet(_) => \"wrong\""
    )
    outer_case = resolved.program.body.items[-1]
    assert isinstance(outer_case, Case)
    assert isinstance(outer_case.branches[0].pattern, ConstructorPattern)
    outer_pattern = outer_case.branches[0].pattern
    outer_candidate = outer_pattern.positional[0]
    assert isinstance(outer_candidate, VarPattern)
    assert isinstance(outer_case.branches[0].body, Block)
    inner_case = outer_case.branches[0].body.items[0]
    assert isinstance(inner_case, Case)
    assert isinstance(inner_case.branches[0].pattern, ConstructorPattern)
    inner_pattern = inner_case.branches[0].pattern
    inner_candidate = inner_pattern.positional[0]
    assert isinstance(inner_candidate, VarPattern)
    slots_by_candidate_ids = {
        tuple(candidate.pattern_node_id for candidate in slot.candidates): slot
        for slot in resolved.pattern_slots.values()
    }
    assert set(slots_by_candidate_ids) == {
        (outer_candidate.node_id,),
        (inner_candidate.node_id,),
    }
    outer_slot = slots_by_candidate_ids[(outer_candidate.node_id,)]
    inner_slot = slots_by_candidate_ids[(inner_candidate.node_id,)]
    assert inner_slot.alternative is not None
    assert inner_slot.alternative.kind is BinderKind.pattern_slot
    assert inner_slot.alternative.slot_id == outer_slot.slot_id
    before_resolution = dict(resolved.resolution)
    before_constructor_refs = dict(resolved.constructor_refs)

    env = TypeEnvironment()
    _TypeBuilder(env).collect(resolved.program)
    checker = _Checker(env, resolved, HostCapabilities())
    selected_slot_ids: set[int] = set()
    select_pattern_slot = checker._select_pattern_slot

    def record_slot_selection(slot: PatternSlot) -> None:
        select_pattern_slot(slot)
        assert slot.slot_id in checker._slot_resolution
        assert slot.slot_id in checker._slot_constructor_refs
        selected_slot_ids.add(slot.slot_id)

    monkeypatch.setattr(checker, "_select_pattern_slot", record_slot_selection)

    with pytest.raises(AglTypeError):
        checker.check_module(resolved.program)

    assert selected_slot_ids == {outer_slot.slot_id, inner_slot.slot_id}
    assert checker._slot_resolution == {}
    assert checker._slot_constructor_refs == {}
    assert resolved.resolution == before_resolution
    assert resolved.constructor_refs == before_constructor_refs


def test_slot_assignment_reports_the_resolved_immutable_binder() -> None:
    resolved = _resolve(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(1)\n"
        "case item of | packet(value) =>\n"
        "  value := 2"
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    assert isinstance(case.branches[0].body, Block)
    assert isinstance(case.branches[0].body.items[0], AssignStmt)

    with pytest.raises(AglTypeError, match="pattern binding"):
        check_module(resolved, HostCapabilities())


def test_checked_module_accessors_pass_through_non_slot_references() -> None:
    checked = check_module(
        _resolve("enum Flag\n  | on\nlet value = 1\nvalue\non"), HostCapabilities()
    )
    value_id, value = next(
        (node_id, ref)
        for node_id, ref in checked.resolved.resolution.items()
        if ref.name == "value"
    )
    ctor_id, ctor = next(
        (node_id, ref) for node_id, ref in checked.resolved.resolution.items() if ref.name == "on"
    )

    assert checked.binding_for(value_id) is value
    assert checked.binding_for(ctor_id) is ctor
    assert checked.constructor_ref_for(ctor_id) is checked.resolved.constructor_refs[ctor_id]


def test_accessors_can_dereference_a_constructed_slot_ref() -> None:
    checked = check_module(_resolve("enum Flag\n  | on\non"), HostCapabilities())
    node_id, binding = next(iter(checked.resolved.resolution.items()))
    constructor = checked.resolved.constructor_refs[node_id]
    slot = BindingRef(
        binding.name,
        False,
        binding.decl_span,
        binding.decl_node_id,
        BinderKind.pattern_slot,
        slot_id=7,
    )
    slotted = replace(
        checked,
        resolved=replace(checked.resolved, resolution={node_id: slot}),
        slot_resolution={7: binding},
        slot_constructor_refs={7: constructor},
    )

    assert slotted.binding_for(node_id) is binding
    assert slotted.constructor_ref_for(node_id) is constructor
