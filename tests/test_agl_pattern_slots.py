"""Data-model contracts for deferred field-directed pattern slots."""

from __future__ import annotations

from dataclasses import replace

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.symbols import BinderKind, BindingRef, PatternSlot, SlotCandidate
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
        (node_id, ref)
        for node_id, ref in checked.resolved.resolution.items()
        if ref.name == "on"
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
