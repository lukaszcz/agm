"""Data-model contracts for deferred field-directed pattern slots."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.symbols import BinderKind, BindingRef, ConstructorRef, PatternSlot, SlotCandidate
from agm.agl.semantics.types import EnumType
from agm.agl.syntax.nodes import (
    AsPattern,
    Block,
    Case,
    ConstructorPattern,
    LetDecl,
    VarPattern,
    VarRef,
)
from agm.agl.typecheck import AglTypeError, TypeEnvironment, check_module

_NESTED_OVERLOADED_FALLBACK_SOURCE = (
    "enum Flag\n"
    "  | on\n"
    "enum Other\n"
    "  | on\n"
    "enum Packet\n"
    "  | packet(flag: Flag)\n"
    "let item = packet(Flag::on)\n"
    "case item of\n"
    "  | packet(on) =>\n"
    "    case item of | packet(on) => on\n"
    "  | packet(_) => 0"
)


def test_pattern_slot_models_construct_and_scope_tables_default_empty() -> None:
    resolved = resolve_module(parse_program("let value = 1\nvalue"))
    alternative = next(iter(resolved.resolution.values()))
    candidate = SlotCandidate(
        pattern_node_id=11,
        span=resolved.program.span,
        unconditional=True,
    )
    slot = PatternSlot(
        slot_id=7,
        name="value",
        candidates=(candidate,),
        alternative=alternative,
        outside_constructor_candidates=2,
    )

    assert slot.slot_id == 7
    assert slot.name == "value"
    assert slot.candidates == (candidate,)
    assert candidate.pattern_node_id == 11
    assert candidate.unconditional
    assert slot.alternative is alternative
    assert slot.outside_constructor_candidates == 2
    assert BinderKind.pattern_slot.value == "pattern_slot"
    assert all(ref.slot_id is None for ref in resolved.resolution.values())
    assert resolved.pattern_slots == {}
    assert resolved.slot_references == {}


def test_scope_emits_one_shared_slot_per_branch_name_in_source_order() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(left: Flag, right: Flag)\n"
            "let item = packet(on(), on())\n"
            "case item of\n"
            "  | packet(on, on) => on\n"
            "  | packet(on, on) => on"
        )
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    branch_patterns = [branch.pattern for branch in case.branches]
    assert all(isinstance(pattern, ConstructorPattern) for pattern in branch_patterns)
    first_pattern, second_pattern = branch_patterns
    assert isinstance(first_pattern, ConstructorPattern)
    assert isinstance(second_pattern, ConstructorPattern)
    first_bares = first_pattern.positional
    second_bares = second_pattern.positional
    assert all(isinstance(pattern, VarPattern) for pattern in (*first_bares, *second_bares))

    assert len(resolved.pattern_slots) == 2
    slots_by_candidates = {
        tuple(candidate.pattern_node_id for candidate in slot.candidates): slot
        for slot in resolved.pattern_slots.values()
    }
    assert set(slots_by_candidates) == {
        tuple(pattern.node_id for pattern in first_bares),
        tuple(pattern.node_id for pattern in second_bares),
    }
    for slot in slots_by_candidates.values():
        assert slot.name == "on"
        assert [candidate.unconditional for candidate in slot.candidates] == [False, False]


def test_scope_slot_alternatives_cover_enclosing_unbound_and_constructor_names() -> None:
    enclosing = resolve_module(
        parse_program(
            "enum Packet\n"
            "  | packet(value: int)\n"
            "let value = 0\n"
            "let item = packet(1)\n"
            "case item of | packet(value) => value"
        )
    )
    unbound = resolve_module(
        parse_program(
            "enum Packet\n"
            "  | packet(value: int)\n"
            "let item = packet(1)\n"
            "case item of | packet(value) => value"
        )
    )
    constructor = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(value: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )

    enclosing_slot = next(iter(enclosing.pattern_slots.values()))
    assert enclosing_slot.alternative is not None
    assert enclosing_slot.alternative.kind is BinderKind.let_binding
    assert enclosing_slot.outside_constructor_candidates == 0

    unbound_slot = next(iter(unbound.pattern_slots.values()))
    assert unbound_slot.alternative is None
    assert unbound_slot.outside_constructor_candidates == 0

    constructor_slot = next(iter(constructor.pattern_slots.values()))
    assert constructor_slot.alternative is not None
    assert constructor_slot.alternative.kind is BinderKind.constructor_binding
    assert constructor_slot.outside_constructor_candidates == 1


def test_scope_slot_alternative_can_reference_an_outer_slot() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Packet\n"
            "  | packet(value: int)\n"
            "let item = packet(1)\n"
            "case item of\n"
            "  | packet(value) =>\n"
            "    case item of\n"
            "      | packet(value) => value"
        )
    )
    outer_case = resolved.program.body.items[-1]
    assert isinstance(outer_case, Case)
    outer_pattern = outer_case.branches[0].pattern
    assert isinstance(outer_pattern, ConstructorPattern)
    outer_bare = outer_pattern.positional[0]
    assert isinstance(outer_bare, VarPattern)
    outer_body = outer_case.branches[0].body
    assert isinstance(outer_body, Block)
    inner_case = outer_body.items[0]
    assert isinstance(inner_case, Case)
    inner_pattern = inner_case.branches[0].pattern
    assert isinstance(inner_pattern, ConstructorPattern)
    inner_bare = inner_pattern.positional[0]
    assert isinstance(inner_bare, VarPattern)

    outer_slot = next(
        slot
        for slot in resolved.pattern_slots.values()
        if slot.candidates[0].pattern_node_id == outer_bare.node_id
    )
    inner_slot = next(
        slot
        for slot in resolved.pattern_slots.values()
        if slot.candidates[0].pattern_node_id == inner_bare.node_id
    )
    assert inner_slot.alternative is not None
    assert inner_slot.alternative.kind is BinderKind.pattern_slot
    assert inner_slot.alternative.slot_id == outer_slot.slot_id


def test_scope_slot_includes_as_candidates_and_bridges_provisional_body_references() -> None:
    resolved = resolve_module(
        parse_program(
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
    first_ref = first.value
    last_ref = body.items[1]

    slot = next(iter(resolved.pattern_slots.values()))
    assert [candidate.pattern_node_id for candidate in slot.candidates] == [
        as_pattern.pattern.node_id,
        as_pattern.node_id,
    ]
    assert [candidate.unconditional for candidate in slot.candidates] == [False, True]
    assert isinstance(first_ref, VarRef)
    assert isinstance(last_ref, VarRef)
    provisional_ids = {candidate.pattern_node_id for candidate in slot.candidates}
    old_reverse_index_refs = {
        node_id
        for node_id, binding in resolved.resolution.items()
        if binding.decl_node_id in provisional_ids
    }
    assert old_reverse_index_refs == {first_ref.node_id, last_ref.node_id}
    assert resolved.slot_references == {node_id: slot.slot_id for node_id in old_reverse_index_refs}


def test_scope_slot_includes_standalone_as_binder_and_bridges_body_reference() -> None:
    resolved = resolve_module(parse_program("let item = 1\ncase item of | _ as value => value"))
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, AsPattern)
    body_ref = case.branches[0].body
    assert isinstance(body_ref, VarRef)

    slot = next(iter(resolved.pattern_slots.values()))
    assert [candidate.pattern_node_id for candidate in slot.candidates] == [pattern.node_id]
    assert [candidate.unconditional for candidate in slot.candidates] == [True]
    assert resolved.resolution[body_ref.node_id].decl_node_id == pattern.node_id
    assert resolved.slot_references == {body_ref.node_id: slot.slot_id}


def test_scope_slot_orders_reversed_as_and_bare_candidates_and_bridges_body_reference() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(left: Flag, right: Flag)\n"
            "let item = packet(on(), on())\n"
            "case item of | packet(_ as on, on) => on"
        )
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    as_pattern, bare_pattern = pattern.positional
    assert isinstance(as_pattern, AsPattern)
    assert isinstance(bare_pattern, VarPattern)
    body_ref = case.branches[0].body
    assert isinstance(body_ref, VarRef)

    slot = next(iter(resolved.pattern_slots.values()))
    assert [candidate.pattern_node_id for candidate in slot.candidates] == [
        as_pattern.node_id,
        bare_pattern.node_id,
    ]
    assert [candidate.unconditional for candidate in slot.candidates] == [True, False]
    assert resolved.resolution[body_ref.node_id].decl_node_id == as_pattern.node_id
    assert resolved.slot_references == {body_ref.node_id: slot.slot_id}


def test_checked_module_accessors_pass_through_raw_scope_tables_without_slots() -> None:
    checked = check_module(
        resolve_module(parse_program("enum Flag\n  | on\nlet value = 1\nvalue\non")),
        HostCapabilities(),
    )
    value_ref_node_id, value_binding = next(
        (node_id, ref)
        for node_id, ref in checked.resolved.resolution.items()
        if ref.name == "value"
    )
    constructor_ref_node_id, constructor_binding = next(
        (node_id, ref) for node_id, ref in checked.resolved.resolution.items() if ref.name == "on"
    )

    assert checked.slot_resolution == {}
    assert checked.slot_constructor_refs == {}
    assert checked.binding_for(value_ref_node_id) is value_binding
    assert checked.binding_for(constructor_ref_node_id) is constructor_binding
    assert (
        checked.constructor_ref_for(constructor_ref_node_id)
        is checked.resolved.constructor_refs[constructor_ref_node_id]
    )
    assert checked.constructor_ref_for(value_ref_node_id) is None


def test_checked_module_accessors_dereference_pattern_slot_binding() -> None:
    checked = check_module(
        resolve_module(parse_program("enum Flag\n  | on\non")), HostCapabilities()
    )
    node_id, constructor_binding = next(iter(checked.resolved.resolution.items()))
    constructor_ref = checked.resolved.constructor_refs[node_id]
    slot_binding = BindingRef(
        name=constructor_binding.name,
        mutable=False,
        decl_span=constructor_binding.decl_span,
        decl_node_id=constructor_binding.decl_node_id,
        kind=BinderKind.pattern_slot,
        slot_id=7,
    )
    slotted = replace(
        checked,
        resolved=replace(checked.resolved, resolution={node_id: slot_binding}),
        slot_resolution={7: constructor_binding},
        slot_constructor_refs={7: constructor_ref},
    )

    assert slotted.binding_for(node_id) is constructor_binding
    assert slotted.constructor_ref_for(node_id) is constructor_ref


def test_typecheck_selects_an_ordinary_pattern_slot_binder() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Packet\n"
            "  | packet(value: int)\n"
            "let item = packet(1)\n"
            "case item of | packet(value) => value"
        )
    )
    checked = check_module(resolved, HostCapabilities())
    slot = next(iter(resolved.pattern_slots.values()))
    body_ref_node_id = next(iter(resolved.slot_references))

    assert checked.slot_resolution[slot.slot_id].decl_node_id == slot.candidates[0].pattern_node_id
    assert slot.slot_id not in checked.slot_constructor_refs
    assert checked.binding_for(body_ref_node_id) == checked.resolved.resolution[body_ref_node_id]
    assert checked.constructor_ref_for(body_ref_node_id) is None


def test_typecheck_selects_a_constructor_pattern_slot() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )
    checked = check_module(resolved, HostCapabilities())
    slot = next(iter(resolved.pattern_slots.values()))
    body_ref_node_id = next(iter(resolved.slot_references))

    assert checked.slot_resolution[slot.slot_id] == checked.resolved.resolution[body_ref_node_id]
    assert (
        checked.slot_constructor_refs[slot.slot_id]
        == checked.resolved.constructor_refs[body_ref_node_id]
    )
    assert checked.binding_for(body_ref_node_id) == checked.resolved.resolution[body_ref_node_id]
    assert (
        checked.constructor_ref_for(body_ref_node_id)
        == checked.resolved.constructor_refs[body_ref_node_id]
    )


def test_typecheck_selects_fallback_alternatives_through_nested_slots() -> None:
    resolved = resolve_module(
        parse_program(
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
    )
    checked = check_module(resolved, HostCapabilities())
    slots = tuple(resolved.pattern_slots.values())
    body_ref_node_id = next(iter(resolved.slot_references))

    assert len(slots) == 2
    assert list(checked.slot_resolution) == [slot.slot_id for slot in slots]
    assert checked.slot_resolution[slots[1].slot_id] == checked.slot_resolution[slots[0].slot_id]
    assert (
        checked.slot_resolution[slots[1].slot_id] == checked.resolved.resolution[body_ref_node_id]
    )
    assert slots[0].slot_id not in checked.slot_constructor_refs
    assert slots[1].slot_id not in checked.slot_constructor_refs
    assert checked.binding_for(body_ref_node_id) == checked.resolved.resolution[body_ref_node_id]


def test_typecheck_slot_selection_reports_the_existing_duplicate_binder_error() -> None:
    source = (
        "enum Packet\n"
        "  | packet(value: int, other: int)\n"
        "let item = packet(1, 2)\n"
        "case item of | packet(value, _ as value) => value"
    )

    with pytest.raises(AglTypeError, match="bound more than once"):
        check_module(resolve_module(parse_program(source)), HostCapabilities())


def test_slot_selector_models_source_ordered_duplicate_with_as_pattern() -> None:
    from agm.agl.typecheck.builder import _TypeBuilder
    from agm.agl.typecheck.checker import _Checker

    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(on: int, other: int)\n"
            "let item = packet(1, 2)\n"
            "case item of | packet(on, _ as on) => on"
        )
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    first_binder, second_binder = pattern.positional
    assert isinstance(first_binder, VarPattern)
    assert isinstance(second_binder, AsPattern)

    env = TypeEnvironment()
    _TypeBuilder(env).collect(resolved.program)
    checker = _Checker(env, resolved, HostCapabilities())
    checker._bind_pattern_types(case.branches[0].pattern, EnumType("Packet"), case.branches[0])

    selections = checker._select_ready_pattern_slots()

    assert len(selections) == 1
    assert selections[0].error is not None
    assert str(selections[0].error) == "Name 'on' is bound more than once in this pattern."
    assert selections[0].error.span == second_binder.span


def test_slot_selector_uses_unique_outside_constructor_candidate() -> None:
    from agm.agl.typecheck.builder import _TypeBuilder
    from agm.agl.typecheck.checker import _Checker

    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    bare = pattern.positional[0]
    assert isinstance(bare, VarPattern)
    slot = next(iter(resolved.pattern_slots.values()))

    env = TypeEnvironment()
    _TypeBuilder(env).collect(resolved.program)
    checker = _Checker(env, resolved, HostCapabilities())
    checker._bind_pattern_types(case.branches[0].pattern, EnumType("Packet"), case.branches[0])
    checker._pattern_classifications[bare.node_id] = ConstructorRef(
        owner_name="unrelated",
        variant="on",
        owner_decl_node_id=-1,
        type_params=(),
    )

    checker._select_ready_pattern_slots()

    assert checker._slot_constructor_refs[slot.slot_id] == resolved.constructor_candidates["on"][0]


def test_slot_selector_models_outside_constructor_ambiguity() -> None:
    from agm.agl.typecheck.builder import _TypeBuilder
    from agm.agl.typecheck.checker import _Checker

    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Other\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(Flag::on)\n"
            "case item of | packet(on) => on"
        )
    )
    case = resolved.program.body.items[-1]
    assert isinstance(case, Case)
    pattern = case.branches[0].pattern
    assert isinstance(pattern, ConstructorPattern)
    bare = pattern.positional[0]
    assert isinstance(bare, VarPattern)

    env = TypeEnvironment()
    _TypeBuilder(env).collect(resolved.program)
    checker = _Checker(env, resolved, HostCapabilities())
    checker._bind_pattern_types(case.branches[0].pattern, EnumType("Packet"), case.branches[0])

    selections = checker._select_ready_pattern_slots()

    assert len(selections) == 1
    assert selections[0].error is not None
    assert str(selections[0].error) == (
        "'on' is ambiguous outside the pattern; qualify the reference."
    )
    assert selections[0].error.span == bare.span
    with pytest.raises(AglTypeError, match="ambiguous outside the pattern"):
        check_module(resolved, HostCapabilities())


def test_self_validation_rejects_shadow_success_when_authoritative_duplicate_fails() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(on: int, other: int)\n"
            "let item = packet(1, 2)\n"
            "case item of | packet(on, _ as on) => on"
        )
    )
    slot = next(iter(resolved.pattern_slots.values()))
    malformed = replace(
        resolved,
        pattern_slots={slot.slot_id: replace(slot, candidates=(slot.candidates[0],))},
    )

    with pytest.raises(AssertionError, match="pattern-slot selection invariant failed"):
        check_module(malformed, HostCapabilities())


def test_self_validation_rejects_different_duplicate_diagnostic() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(on: int, other: int)\n"
            "let item = packet(1, 2)\n"
            "case item of | packet(on, _ as on) => on"
        )
    )
    slot = next(iter(resolved.pattern_slots.values()))
    malformed = replace(
        resolved,
        pattern_slots={slot.slot_id: replace(slot, candidates=tuple(reversed(slot.candidates)))},
    )

    with pytest.raises(AssertionError, match="selection diagnostics disagree"):
        check_module(malformed, HostCapabilities())


def test_self_validation_rejects_shadow_ambiguity_when_authority_succeeds() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )
    slot = next(iter(resolved.pattern_slots.values()))
    malformed = replace(
        resolved,
        pattern_slots={slot.slot_id: replace(slot, outside_constructor_candidates=2)},
    )

    with pytest.raises(AssertionError, match="shadow selection failed"):
        check_module(malformed, HostCapabilities())


def test_nested_overloaded_fallback_keeps_authoritative_error_with_self_validation() -> None:
    with pytest.raises(AglTypeError):
        check_module(
            resolve_module(parse_program(_NESTED_OVERLOADED_FALLBACK_SOURCE)), HostCapabilities()
        )


def test_nested_overloaded_fallback_keeps_authoritative_error_without_self_validation(
    self_validation_disabled: None,
) -> None:
    with pytest.raises(AglTypeError):
        check_module(
            resolve_module(parse_program(_NESTED_OVERLOADED_FALLBACK_SOURCE)), HostCapabilities()
        )


def test_production_path_keeps_authoritative_outside_ambiguity(
    self_validation_disabled: None,
) -> None:
    source = (
        "enum Flag\n"
        "  | on\n"
        "enum Other\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(Flag::on)\n"
        "case item of | packet(on) => on"
    )

    with pytest.raises(AglTypeError, match="ambiguous outside the pattern"):
        check_module(resolve_module(parse_program(source)), HostCapabilities())


def test_typecheck_rejects_a_binderless_slot_without_an_alternative() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )
    slot = next(iter(resolved.pattern_slots.values()))
    malformed = replace(
        resolved,
        pattern_slots={slot.slot_id: replace(slot, alternative=None)},
    )

    with pytest.raises(AssertionError):
        check_module(malformed, HostCapabilities())


def test_typecheck_self_validation_rejects_slot_reconciliation_mismatch() -> None:
    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "let value = 0\n"
            "let item = packet(on())\n"
            "case item of | packet(on) => on"
        )
    )
    slot = next(iter(resolved.pattern_slots.values()))
    value = resolved.program.body.items[2]
    assert isinstance(value, LetDecl)
    malformed_alternative = BindingRef(
        name="on",
        mutable=False,
        decl_span=value.span,
        decl_node_id=value.node_id,
        kind=BinderKind.let_binding,
    )
    malformed = replace(
        resolved,
        pattern_slots={slot.slot_id: replace(slot, alternative=malformed_alternative)},
    )

    with pytest.raises(AssertionError):
        check_module(malformed, HostCapabilities())


def test_typecheck_rolls_back_selected_nested_pattern_slots() -> None:
    from agm.agl.typecheck.builder import _TypeBuilder
    from agm.agl.typecheck.checker import _Checker

    resolved = resolve_module(
        parse_program(
            "enum Flag\n"
            "  | on\n"
            "enum Packet\n"
            "  | packet(flag: Flag)\n"
            "var on: int = 0\n"
            "let item = packet(Flag::on)\n"
            "case item of\n"
            "  | packet(on) =>\n"
            "    case item of\n"
            "      | packet(on) =>\n"
            '        on := "wrong"\n'
            "        on\n"
            "      | packet(_) => 0\n"
            "  | packet(_) => 0"
        )
    )
    env = TypeEnvironment()
    _TypeBuilder(env).collect(resolved.program)
    checker = _Checker(env, resolved, HostCapabilities())

    with pytest.raises(AglTypeError):
        checker.check_module(resolved.program)

    assert checker._slot_resolution == {}
    assert checker._slot_constructor_refs == {}
    assert checker._selected_pattern_slots == {}


def test_checked_module_accessors_support_slot_reference_table() -> None:
    checked = check_module(
        resolve_module(parse_program("enum Flag\n  | on\non")), HostCapabilities()
    )
    node_id, constructor_binding = next(iter(checked.resolved.resolution.items()))
    constructor_ref = checked.resolved.constructor_refs[node_id]
    slotted = replace(
        checked,
        resolved=replace(checked.resolved, slot_references={node_id: 7}),
        slot_resolution={7: constructor_binding},
        slot_constructor_refs={7: constructor_ref},
    )

    assert slotted.binding_for(node_id) is constructor_binding
    assert slotted.constructor_ref_for(node_id) is constructor_ref
