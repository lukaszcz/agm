"""Field-directed pattern slots: what programs mean, and which are rejected.

A slot is a branch-local binding shared by every occurrence of one name in a
case pattern; typechecking decides whether it finally denotes a binder or a
constructor.  These tests exercise that decision through program behaviour and
through the checked artifact's public accessors, which are what consumers use.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from agm.agl import PipelineDriver
from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.symbols import AglScopeError, BinderKind
from agm.agl.typecheck import AglTypeError, check_module


def _resolve(source: str):
    return resolve_module(parse_program(source))


def _run(source: str) -> tuple[bool, str, list[str]]:
    """Run *source*, returning its success flag, stdout, and diagnostics."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = PipelineDriver().run(source, param_values={})
    return result.ok, buffer.getvalue(), [d.message for d in result.diagnostics]


def _slot_reference(resolved) -> int:
    """The node id of the sole branch-body reference that resolves to a slot."""
    return next(
        node_id
        for node_id, ref in resolved.resolution.items()
        if ref.kind is BinderKind.pattern_slot
    )


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_each_branch_gets_its_own_slot_for_a_repeated_name() -> None:
    ok, out, diagnostics = _run(
        "enum Flag\n"
        "  | on\n"
        "  | off\n"
        "enum Packet\n"
        "  | packet(left: Flag, right: Flag)\n"
        "let item = packet(Flag::on, Flag::on)\n"
        "print(case item of\n"
        '  | packet(on, on) => "both on"\n'
        '  | packet(_, _) => "other")\n'
    )

    assert ok, diagnostics
    assert out == "both on\n"


def test_every_branch_body_reference_sees_the_as_pattern_binding() -> None:
    ok, out, diagnostics = _run(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(7)\n"
        "print(case item of\n"
        "  | packet(_ as value) => value + value)\n"
    )

    assert ok, diagnostics
    assert out == "14\n"


def test_an_inner_slot_shadows_the_enclosing_one_of_the_same_name() -> None:
    ok, out, diagnostics = _run(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(3)\n"
        "let result = case item of\n"
        "  | packet(value) =>\n"
        "    case item of\n"
        "      | packet(value) => value * 10\n"
        "print result\n"
    )

    assert ok, diagnostics
    assert out == "30\n"


def test_a_nested_constructor_slot_falls_back_to_the_enclosing_binding() -> None:
    ok, out, diagnostics = _run(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "var on: int = 4\n"
        "let item = packet(Flag::on)\n"
        "let result = case item of\n"
        "  | packet(on) =>\n"
        "    case item of\n"
        "      | packet(on) => on\n"
        "print result\n"
    )

    assert ok, diagnostics
    assert out == "4\n"


def test_a_slot_selected_as_a_constructor_is_callable_in_the_branch_body() -> None:
    ok, out, diagnostics = _run(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(on())\n"
        "print(case item of | packet(on) => on() == Flag::on)\n"
    )

    assert ok, diagnostics
    assert out == "true\n"


def test_constructor_selected_slot_preserves_agent_usage() -> None:
    resolved = _resolve(
        "agent on\n"
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(Flag::on)\n"
        "case item of\n"
        '  | packet(on) => ask("question", agent = on)\n'
    )

    checked = check_module(
        resolved,
        HostCapabilities(
            agent_names=frozenset({"on"}),
            has_default_agent=True,
            codec_kinds={"text": frozenset({"text"})},
        ),
    )

    slot_reference = _slot_reference(resolved)
    binding = checked.binding_for(slot_reference)
    assert binding is not None
    assert binding.kind is BinderKind.agent_binding
    assert resolved.warnings == ()


def test_a_bare_nullary_variant_name_tests_the_variant() -> None:
    ok, out, diagnostics = _run(
        "enum Flag\n"
        "  | on\n"
        "  | off\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(Flag::off)\n"
        "print(case item of | packet(on) => 1 | packet(_) => 2)\n"
    )

    assert ok, diagnostics
    assert out == "2\n"


@pytest.mark.parametrize(
    ("declaration", "name"),
    (
        ("record Token\n  value: int\n", "Token"),
        ("enum Other\n  | payload(v: int)\n", "payload"),
    ),
)
def test_a_bare_name_that_cannot_match_nullary_is_rejected(declaration: str, name: str) -> None:
    ok, _out, diagnostics = _run(
        f"{declaration}"
        "enum Packet\n"
        "  | packet(left: int)\n"
        "let item = packet(1)\n"
        f"print(case item of | packet({name}) => 1)\n"
    )

    assert not ok
    assert name in diagnostics[0]


def test_assigning_to_a_slot_selected_as_a_binder_is_rejected() -> None:
    ok, _out, diagnostics = _run(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(1)\n"
        "case item of | packet(value) =>\n"
        "  value := 2\n"
    )

    assert not ok
    assert "pattern binding" in diagnostics[0]


def test_ambiguous_slot_assignment_is_diagnosed_at_the_target() -> None:
    resolved = _resolve(
        "enum Color\n"
        "  | Red\n"
        "  | Blue\n"
        "enum Signal\n"
        "  | Red\n"
        "  | Green\n"
        "enum Wrap\n"
        "  | wrap(shade: Color)\n"
        "let item = wrap(Color::Blue)\n"
        "case item of\n"
        "  | wrap(Red) =>\n"
        "    Red := Color::Blue\n"
    )

    with pytest.raises(AglTypeError) as exc_info:
        check_module(resolved, HostCapabilities())

    diagnostic = exc_info.value.to_diagnostic()
    assert diagnostic.line == 12


# ---------------------------------------------------------------------------
# Duplicate binders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ("packet(value, _ as value)", "packet(_ as value, value)"))
def test_typecheck_diagnoses_duplicate_binders_that_could_have_been_variants(
    pattern: str,
) -> None:
    """A spelling that could name a nullary variant stays undecided until checking."""
    resolved = _resolve(
        "enum Flag\n"
        "  | value\n"
        "enum Packet\n"
        "  | packet(value: int, other: int)\n"
        "let item = packet(1, 2)\n"
        f"case item of | {pattern} => value"
    )

    with pytest.raises(AglTypeError):
        check_module(resolved, HostCapabilities())


@pytest.mark.parametrize(
    ("declaration", "name"),
    (
        ("enum Flag\n  | value(data: int)\n", "value"),
        ("record Token\n  value: int\n", "Token"),
        ("", "Retry"),
        ("", "ExecResult"),
    ),
)
@pytest.mark.parametrize("order", ("{n}, _ as {n}", "_ as {n}, {n}"))
def test_scope_rejects_duplicate_binders_that_could_not_be_variants(
    declaration: str, name: str, order: str
) -> None:
    """No spelling in the slot can be a nullary variant, so scope decides at once."""
    positions = order.format(n=name)
    with pytest.raises(AglScopeError):
        _resolve(
            f"{declaration}"
            "enum Packet\n"
            "  | packet(left: int, right: int)\n"
            "let item = packet(1, 2)\n"
            f"case item of | packet({positions}) => {name}"
        )


# ---------------------------------------------------------------------------
# Checked-artifact accessors
# ---------------------------------------------------------------------------


def test_accessors_dereference_a_slot_selected_as_a_binder() -> None:
    resolved = _resolve(
        "enum Packet\n"
        "  | packet(value: int)\n"
        "let item = packet(1)\n"
        "case item of | packet(value) => value"
    )
    reference = _slot_reference(resolved)

    checked = check_module(resolved, HostCapabilities())

    binding = checked.binding_for(reference)
    assert binding is not None
    assert binding.kind is BinderKind.pattern_binding
    assert checked.constructor_ref_for(reference) is None


def test_accessors_dereference_a_slot_selected_as_a_constructor() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "let item = packet(on())\n"
        "case item of | packet(on) => on"
    )
    reference = _slot_reference(resolved)

    checked = check_module(resolved, HostCapabilities())

    binding = checked.binding_for(reference)
    assert binding is not None
    assert binding.kind is BinderKind.constructor_binding
    assert checked.constructor_ref_for(reference) is not None


def test_accessors_dereference_a_nested_slot_to_the_enclosing_binding() -> None:
    resolved = _resolve(
        "enum Flag\n"
        "  | on\n"
        "enum Packet\n"
        "  | packet(flag: Flag)\n"
        "var on: int = 0\n"
        "let item = packet(Flag::on)\n"
        "case item of\n"
        "  | packet(on) =>\n"
        "    case item of | packet(on) => on"
    )
    reference = _slot_reference(resolved)

    checked = check_module(resolved, HostCapabilities())

    binding = checked.binding_for(reference)
    assert binding is not None
    assert binding.kind is BinderKind.var_binding
    assert checked.constructor_ref_for(reference) is None


def test_accessors_pass_through_references_that_are_not_slots() -> None:
    checked = check_module(
        _resolve("enum Flag\n  | on\nlet value = 1\nlet _ = value\non"), HostCapabilities()
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


def test_checking_leaves_scope_output_untouched_and_stays_repeatable() -> None:
    """Rejected checking must not leak speculative selections into its input."""
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
        '  | packet(_) => "wrong"'
    )
    before_resolution = dict(resolved.resolution)
    before_constructor_refs = dict(resolved.constructor_refs)

    with pytest.raises(AglTypeError) as first:
        check_module(resolved, HostCapabilities())
    with pytest.raises(AglTypeError) as second:
        check_module(resolved, HostCapabilities())

    assert str(first.value) == str(second.value)
    assert resolved.resolution == before_resolution
    assert resolved.constructor_refs == before_constructor_refs
