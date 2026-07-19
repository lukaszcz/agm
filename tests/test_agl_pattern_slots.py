"""Data-model contracts for deferred field-directed pattern slots."""

from __future__ import annotations

from dataclasses import replace

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.symbols import BinderKind, BindingRef, PatternSlot, SlotCandidate
from agm.agl.syntax.nodes import (
    AsPattern,
    Block,
    Case,
    ConstructorPattern,
    LetDecl,
    VarPattern,
    VarRef,
)
from agm.agl.typecheck import check_module


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
    resolved = resolve_module(
        parse_program(
            "let item = 1\n"
            "case item of | _ as value => value"
        )
    )
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
