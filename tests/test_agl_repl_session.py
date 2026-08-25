"""Tests for the UI-free REPL session core (``agm.agl.repl.session``).

Drives ``ReplSession`` directly with source strings and fake agents.  Asserts
user-visible behaviour: persistence across entries, redefinition/shadowing,
expression/binding echo data, ``type_of`` purity, partial effects on failure,
exactly-once agent dispatch, the ``:set`` param flow, ``reset``, ``load_file``,
``dump_source``, surfaced warnings, and ``check_only`` (type-only) runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import pytest

from agm.agl.diagnostics import AglError
from agm.agl.ir.program import IrParam
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS, create_seeded_type_table
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    BoolType,
    BottomType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
)
from agm.agl.semantics.values import (
    VOID_VALUE,
    ArrayValue,
    BoolValue,
    IntValue,
    RecordValue,
    TextValue,
    UnitValue,
)
from tests._agl_helpers import strip_decl_ids
from tests._process_helpers import FakeShell

# ---------------------------------------------------------------------------
# Fake agents
# ---------------------------------------------------------------------------


class CountingAgent:
    """A fake ``AgentFn`` that counts invocations and returns scripted replies."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or ["ok"]
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        idx = min(self.calls, len(self._responses) - 1)
        reply = self._responses[idx]
        self.calls += 1
        return AgentResponse(content=reply)


def _literal_for_type(typ: Type) -> str:
    if isinstance(typ, TextType):
        return '"x"'
    if isinstance(typ, IntType):
        return "1"
    if isinstance(typ, BoolType):
        return "false"
    if isinstance(typ, JsonType):
        return "{}"
    if isinstance(typ, EnumType) and typ.name == "Option":
        return "None"
    if isinstance(typ, EnumType) and typ.name == "Agent":
        return 'AgentCommand("x")'
    raise AssertionError(f"no test literal for {typ!r}")


def _constructor_args(fields: dict[str, Type]) -> str:
    return ", ".join(f"{name} = {_literal_for_type(typ)}" for name, typ in fields.items())


# ---------------------------------------------------------------------------
# Persistence across entries
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_scope_only_entry_is_a_declaration_without_an_initializer(self) -> None:
        result = ReplSession().eval_entry("scope A\nend A")

        assert result.ok, result.diagnostics
        assert result.kind == "declaration"
        assert result.value is None

    def test_scoped_type_persists_with_a_same_named_root_type(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nrecord Token()\nend A").ok
        assert s.eval_entry("record Token()").ok

        scoped = s.eval_entry("let token: A::Token = A::Token()")
        root = s.eval_entry("let token: Token = Token()")
        mismatch = s.eval_entry("let token: Token = A::Token()")

        assert scoped.ok, scoped.diagnostics
        assert root.ok, root.diagnostics
        assert not mismatch.ok

    def test_later_entry_extends_a_retained_scope_with_a_generic_type(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nend A").ok
        assert s.eval_entry("scope A\nrecord Box[T]\n  value: T\nend A").ok

        result = s.eval_entry("let box: A::Box[int] = A::Box(value = 1)")

        assert result.ok, result.diagnostics

    def test_builtin_agent_method_is_callable_across_entries(self) -> None:
        agent = CountingAgent("42")
        session = ReplSession(agent_dispatcher=agent)
        assert session.eval_entry('let worker: Agent = AgentCommand("worker")').ok

        result = session.eval_entry('worker.ask::[int]("How many?")')

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)
        assert agent.calls == 1

    def test_shell_timeout_seed_matches_loaded_option_members(self) -> None:
        """A REPL timeout seed remains matchable after loading ``std/core``."""
        session = ReplSession(shell_exec_timeout=2.0)

        result = session.eval_entry(
            "import std/config\n"
            "case std/config::timeout of\n"
            "  | Some(value) => value\n"
            '  | None => "disabled"\n'
        )

        assert result.ok, result.diagnostics
        assert result.value == TextValue("2.0s")

    def test_method_declared_after_its_type_is_callable_in_a_later_entry(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Meter(value: int)").ok
        assert session.eval_entry(
            "def Meter::add(self, amount: int) -> int = self.value + amount"
        ).ok

        result = session.eval_entry("Meter(value = 40).add(2)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_later_method_cannot_collide_with_a_retained_owner_field(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Meter(value: int)").ok

        rejected = session.eval_entry("def Meter::value(self) -> int = 0")

        assert not rejected.ok
        message = rejected.diagnostics[0].message.lower()
        assert "value" in message
        assert "field" in message
        assert "method" in message

    def test_later_base_method_cannot_collide_with_a_retained_descendant_member(self) -> None:
        session = ReplSession()
        assert session.eval_entry("exception Base extends Exception\n  code: int").ok
        assert session.eval_entry("exception Child extends Base\n  detail: text").ok
        assert session.eval_entry('def Child::label(self) -> text = "child"').ok

        rejected = session.eval_entry('def Base::label(self) -> text = "base"')

        assert not rejected.ok
        message = rejected.diagnostics[0].message.lower()
        assert "label" in message
        assert "method" in message

    def test_later_base_method_cannot_collide_with_a_retained_descendant_field(self) -> None:
        session = ReplSession()
        assert session.eval_entry("exception Base extends Exception\n  code: int").ok
        assert session.eval_entry("exception Child extends Base\n  detail: text").ok

        rejected = session.eval_entry('def Base::detail(self) -> text = "base"')

        assert not rejected.ok
        message = rejected.diagnostics[0].message.lower()
        assert "detail" in message
        assert "field" in message

    def test_bound_method_binding_persists_across_entries(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Meter(value: int)").ok
        assert session.eval_entry(
            "def Meter::add(self, amount: int) -> int = self.value + amount"
        ).ok
        assert session.eval_entry("let add = Meter(value = 40).add").ok

        result = session.eval_entry("add(2)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_exception_method_declared_in_a_later_entry_is_callable(self) -> None:
        session = ReplSession()
        assert session.eval_entry("exception Fault extends Exception\n  code: int").ok
        assert session.eval_entry('def Fault::label(self) -> text = "fault %{self.code}"').ok

        result = session.eval_entry('Fault(message = "bad", code = 7).label()')

        assert result.ok, result.diagnostics
        assert result.value == TextValue("fault 7")

    def test_generic_receiver_method_declared_in_a_later_entry_is_callable(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Box[T](value: T)").ok
        assert session.eval_entry("def Box::get[T](self) -> T = self.value").ok

        result = session.eval_entry("Box(value = 42).get()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_redeclaring_a_method_replaces_its_prior_member_entry(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Meter(value: int)").ok
        assert session.eval_entry("def Meter::read(self) -> int = self.value").ok
        assert session.eval_entry("def Meter::read(self) -> int = self.value + 1").ok

        result = session.eval_entry("Meter(value = 41).read()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_retained_alias_scope_rejects_a_method_with_its_structural_target(self) -> None:
        session = ReplSession()
        assert session.eval_entry("type Callback = (int) -> bool").ok

        rejected = session.eval_entry("def Callback::bad(self) -> int = 1")

        assert not rejected.ok
        message = rejected.diagnostics[0].message
        assert "alias scope" in message
        assert "Callback" in message
        assert "int -> bool" in message

    def test_retained_alias_redeclared_as_a_record_accepts_a_method_in_a_later_entry(
        self,
    ) -> None:
        session = ReplSession()
        assert session.eval_entry("type Alias = int").ok
        assert session.eval_entry("record Alias\n  x: int\ndef Alias::m(self) -> int = self.x").ok

        result = session.eval_entry("Alias(x = 1).m()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_record_redeclared_as_an_alias_still_rejects_a_method(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Alias\n  x: int").ok

        rejected = session.eval_entry("type Alias = int\ndef Alias::m(self) -> int = 1")

        assert not rejected.ok
        message = rejected.diagnostics[0].message
        assert "alias scope" in message
        assert "Alias" in message
        assert "int" in message

    def test_method_named_after_a_builtin_call_does_not_shadow_the_builtin(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = ReplSession()
        assert session.eval_entry(
            "record Point\n  x: int\n"
            "def Point::print(self) -> int =\n"
            '  print("x = %{self.x}")\n'
            "  self.x"
        ).ok

        result = session.eval_entry("Point(x = 7).print()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(7)
        assert capsys.readouterr().out.strip() == "x = 7"

    def test_scope_region_must_close_in_the_same_entry(self) -> None:
        s = ReplSession()

        unclosed = s.eval_entry("scope Point\ndef distance() -> int = 1")
        stray_closer = s.eval_entry("end Point")
        ordinary_entry = s.eval_entry("let distance = 1")

        assert not unclosed.ok
        assert not stray_closer.ok
        assert ordinary_entry.ok, ordinary_entry.diagnostics

    def test_scoped_members_accumulate_by_path_across_block_and_shorthand_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Shape::area() -> int = 1").ok
        assert s.eval_entry("scope Shape\ndef perimeter() -> int = 2\nend Shape").ok

        result = s.eval_entry("Shape::area() + Shape::perimeter()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(3)

    def test_replacing_scoped_member_keeps_siblings_at_the_same_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry(
            "scope Shape\ndef area() -> int = 1\ndef perimeter() -> int = 2\nend Shape"
        ).ok
        assert s.eval_entry("def Shape::area() -> int = 3").ok

        result = s.eval_entry("Shape::area() + Shape::perimeter()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(5)

    def test_replacing_one_scoped_path_does_not_replace_a_same_named_sibling_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Left::measure() -> int = 1").ok
        assert s.eval_entry("def Right::measure() -> int = 2").ok
        assert s.eval_entry("scope Left\ndef measure() -> int = 3\nend Left").ok

        result = s.eval_entry("Left::measure() + Right::measure()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(5)

    def test_use_exposes_scope_members_in_later_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope Tools\ndef twice(x: int) -> int = x * 2\nend Tools").ok
        assert s.eval_entry("use Tools::*").ok

        result = s.eval_entry("twice(3)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(6)

    def test_use_exposes_scope_members_in_later_scope_extensions(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Source::value() -> int = 2").ok
        assert s.eval_entry("scope Target\nuse Source::*\nend Target").ok
        assert s.eval_entry("scope Target\ndef doubled() -> int = value() * 2\nend Target").ok

        result = s.eval_entry("Target::doubled()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(4)

    def test_type_of_scoped_record_displays_its_qualified_name(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope Geometry\nrecord Point(x: int)\nend Geometry").ok

        assert s.type_of("Geometry::Point(x = 1)") == "record Geometry::Point\n  x: int"

    def test_simple_let_uses_one_frame_slot_per_entry(self) -> None:
        session = ReplSession()

        first = session.eval_entry("let value = 1")
        assert first.ok, first.diagnostics
        assert len(session._ir_base_frame) == 1
        assert session.bindings() == [("value", IntType(), IntValue(1))]

        second = session.eval_entry("let value = 2")
        assert second.ok, second.diagnostics
        assert len(session._ir_base_frame) == 2
        assert session.bindings() == [("value", IntType(), IntValue(2))]

    def test_lambda_binding_initializes_repl_parameter_default(self) -> None:
        session = ReplSession()

        result = session.eval_entry("let f = fn() -> int => 7\nparam value: int = f()")

        assert result.ok, result.diagnostics
        bindings = {name: value for name, _type, value in session.bindings()}
        assert set(bindings) == {"f", "value"}
        assert bindings["value"] == IntValue(7)
        assert session.eval_entry("f()").value == IntValue(7)

    def test_top_level_return_rejected_and_session_continues(self) -> None:
        s = ReplSession()
        bad = s.eval_entry("return 1")
        assert not bad.ok
        assert bad.diagnostics
        assert "return" in bad.diagnostics[0].message.lower()
        good = s.eval_entry("let x = 1")
        assert good.ok, good.diagnostics
        later = s.eval_entry("x")
        assert later.ok, later.diagnostics
        assert later.value == IntValue(1)

    def test_binding_persists_into_next_entry(self) -> None:
        s = ReplSession()
        r1 = s.eval_entry("let x = 1 + 2")
        assert r1.ok
        r2 = s.eval_entry("let y = x * 10")
        assert r2.ok
        names = {n: v for n, _t, v in s.bindings()}
        assert {"x", "y"} <= set(names)

    def test_graph_loader_agl_error_retains_related_notes(self) -> None:
        from unittest.mock import patch

        from agm.agl.syntax.spans import SourceSpan

        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("load failed", related=(("constraint", related),))
        with patch("agm.agl.modules.loader.build_repl_graph", side_effect=error):
            result = ReplSession().eval_entry("1")

        assert not result.ok
        assert result.diagnostics[0].related[0].message == "constraint"

    def test_node_ids_advance_across_entries(self) -> None:
        # Two entries that each declare a distinct binding must both survive —
        # which only works if node ids stay globally unique (binding-type table
        # is keyed by decl node id).
        s = ReplSession()
        s.eval_entry("let a = 1")
        s.eval_entry("let b = 2")
        vals = {n: v for n, _t, v in s.bindings()}
        assert vals["a"] != vals["b"]

    def test_expression_reads_prior_binding(self) -> None:
        s = ReplSession()
        s.eval_entry("let n = 7")
        r = s.eval_entry("n + 1")
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 8

    def test_later_entry_mutates_array_bound_in_earlier_entry(self) -> None:
        """Arrays are mutable reference values, so a later entry's indexed
        assignment through the same binding is visible when that binding is
        read again -- the REPL's cross-entry persistence shares the array
        object, not a copy of it."""
        s = ReplSession()
        assert s.eval_entry("var xs = [1, 2]").ok

        assign = s.eval_entry("xs[0] := 9")
        result = s.eval_entry("xs")

        assert assign.ok, assign.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == ArrayValue([IntValue(9), IntValue(2)])

    def test_echoing_a_value_that_turned_cyclic_since_its_entry_reports_instead_of_crashing(
        self,
    ) -> None:
        """A binding can become cyclic through a *later* entry's indexed
        assignment. Echoing it back afterward runs outside any `try`/`catch`
        scope (the entry that would raise already succeeded), so the REPL
        reports the same one-line message a runtime raise would rather than
        crashing the session."""
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        assert s.eval_entry("record Node(children: array[Node])").ok
        assert s.eval_entry("var xs: array[Node] = [Node(children = [])]").ok
        assert s.eval_entry("let n = Node(children = xs)").ok
        assert s.eval_entry("xs[0] := n").ok

        echoed = s.eval_entry("xs")

        assert echoed.ok, echoed.diagnostics
        rendered = render_entry_result(echoed, echo=True)
        assert rendered is not None
        assert "CyclicValueError" in rendered

    def test_assign_to_prior_immutable_binding_is_rejected(self) -> None:
        """A later entry cannot reassign an earlier ``let``, and it survives."""
        s = ReplSession()
        assert s.eval_entry("let k = 1").ok

        result = s.eval_entry("k := 2")

        assert not result.ok
        assert "cannot assign" in result.diagnostics[0].message.lower()
        assert _int(dict((n, v) for n, _t, v in s.bindings())["k"]) == 1

    def test_new_constructor_does_not_shadow_persisted_value(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let on = 7").ok

        result = s.eval_entry(
            "enum Flag\n  | on\nlet matched = case Flag::on of\n  | on => 8\non + matched"
        )

        assert result.ok, result.diagnostics
        assert result.value == IntValue(15)

    def test_new_record_colliding_with_persisted_binding_is_rejected(self) -> None:
        """A record's constructor name is its root declaration, so nothing else can claim it."""
        s = ReplSession()
        assert s.eval_entry("let Widget = 1").ok

        result = s.eval_entry("let other = 2\nrecord Widget\n  x: int")

        assert not result.ok
        assert "already declared" in result.diagnostics[0].message.lower()
        # The complaint locates the colliding declaration, not a placeholder line.
        assert result.diagnostics[0].line == 2

    def test_selected_pattern_slots_lower_in_repl_entries(self) -> None:
        s = ReplSession()

        binder_result = s.eval_entry(
            "enum Flag\n  | on\n"
            "enum Packet\n  | packet(flag: Flag)\n"
            "var on = 1\nlet item = packet(Flag::on)\n"
            "let result = case item of | packet(on) =>\n"
            "  on := on + 1\n"
            "  on\n"
            "result"
        )
        constructor_session = ReplSession()
        constructor_result = constructor_session.eval_entry(
            "enum Flag\n  | on\n"
            "enum Packet\n  | packet(flag: Flag)\n"
            "let item = packet(Flag::on)\n"
            "case item of | packet(on) => on() == Flag::on"
        )

        assert binder_result.ok, binder_result.diagnostics
        assert binder_result.value == IntValue(2)
        assert constructor_result.ok, constructor_result.diagnostics
        assert constructor_result.value == BoolValue(True)

    def test_program_func_def_is_callable_across_entries(self) -> None:
        session = ReplSession()

        declared = session.eval_entry("program def answer() -> unit = ()")
        result = session.eval_entry("answer()")

        assert declared.ok, declared.diagnostics
        assert declared.kind == "declaration"
        assert result.ok, result.diagnostics
        assert result.value == UnitValue()

    def test_partial_application_closure_persists_into_next_entry(self) -> None:
        s = ReplSession()

        r1 = s.eval_entry("def add(x: int, y: int) -> int = x + y\nlet add1 = add(1, ?)")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("add1(2)")

        assert r2.ok, r2.diagnostics
        assert r2.kind == "expression"
        assert r2.value is not None
        assert _int(r2.value) == 3

    def test_recursive_candidate_signature_promotes_only_after_validation(self) -> None:
        s = ReplSession()

        defined = s.eval_entry(
            "def fib(n: int) = if n < 2 => n else => fib(n - 1) + fib(n - 2)\nfib(8)"
        )
        later = s.eval_entry("fib(10)")

        assert defined.ok, defined.diagnostics
        assert defined.value == IntValue(21)
        assert later.ok, later.diagnostics
        assert later.value == IntValue(55)
        assert s.type_of("fib") == "int -> int"

    def test_partial_promotion_tracks_lazy_initializers_by_identity(self) -> None:
        session = ReplSession()

        failed = session.eval_entry(
            "let broken: decimal = 1 / 0\nlet later = 7\nparam p: int = later"
        )

        assert not failed.ok
        bindings = {name: value for name, _type, value in session.bindings()}
        assert bindings == {"later": IntValue(7)}
        assert session.eval_entry("later").value == IntValue(7)
        assert not session.eval_entry("broken").ok
        assert not session.eval_entry("p").ok

    def test_failed_recursive_candidate_entry_promotes_nothing(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let stable = 1").ok

        failed = s.eval_entry("let transient = 2\ndef loop() = loop()")
        stable = s.eval_entry("stable")
        transient = s.eval_entry("transient")
        loop = s.eval_entry("loop()")

        assert not failed.ok
        assert stable.ok, stable.diagnostics
        assert stable.value == IntValue(1)
        assert not transient.ok
        assert not loop.ok
        assert {name for name, _typ, _value in s.bindings()} == {"stable"}


# ---------------------------------------------------------------------------
# Scoped binding retention
# ---------------------------------------------------------------------------


class TestScopedBindingRetention:
    """A scoped ``let``/``var``/``param`` retains across REPL entries.

    Mirrors how scoped declarations (``def``, types) already retain by path
    atom in ``TestPersistence`` above: a same-path binding declared later
    replaces the retained one, region and shorthand spellings retain
    identically, and a same-entry duplicate is still an error.
    """

    def test_region_form_binding_visible_bare_and_by_path_in_a_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nlet x = 1\nend A").ok

        bare = s.eval_entry("scope A\ndef read() -> int = x\nend A")
        by_path = s.eval_entry("A::x")

        assert bare.ok, bare.diagnostics
        assert by_path.ok, by_path.diagnostics
        assert by_path.value == IntValue(1)
        call = s.eval_entry("A::read()")
        assert call.ok, call.diagnostics
        assert call.value == IntValue(1)

    def test_shorthand_form_binding_visible_bare_and_by_path_in_a_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::y = 2").ok

        bare = s.eval_entry("scope A\ndef read() -> int = y\nend A")
        by_path = s.eval_entry("A::y")

        assert bare.ok, bare.diagnostics
        assert by_path.ok, by_path.diagnostics
        assert by_path.value == IntValue(2)
        call = s.eval_entry("A::read()")
        assert call.ok, call.diagnostics
        assert call.value == IntValue(2)

    def test_region_and_shorthand_forms_retain_identically(self) -> None:
        region = ReplSession()
        assert region.eval_entry("scope A\nlet x = 1\nend A").ok
        region_result = region.eval_entry("A::x")

        shorthand = ReplSession()
        assert shorthand.eval_entry("let A::x = 1").ok
        shorthand_result = shorthand.eval_entry("A::x")

        assert region_result.ok, region_result.diagnostics
        assert shorthand_result.ok, shorthand_result.diagnostics
        assert region_result.value == shorthand_result.value == IntValue(1)

    def test_redeclaring_a_retained_scoped_binding_replaces_it(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok

        replaced = s.eval_entry("let A::x = 99")
        result = s.eval_entry("A::x")

        assert replaced.ok, replaced.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(99)

    def test_same_entry_duplicate_scoped_binding_is_still_an_error(self) -> None:
        s = ReplSession()

        result = s.eval_entry("scope A\nlet z = 1\nlet z = 2\nend A")

        assert not result.ok

    def test_same_entry_duplicate_still_errors_after_a_prior_entry_retained_it(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok

        result = s.eval_entry("scope A\nlet z = 2\nlet z = 3\nend A")

        assert not result.ok

    def test_shorthand_scoped_let_does_not_leak_into_the_bare_root_name(self) -> None:
        """A shorthand ``let A::x`` must not promote as a root binding named ``x``."""
        s = ReplSession()
        assert s.eval_entry("let A::x = 2").ok

        bare = s.eval_entry("x")

        assert not bare.ok

    def test_shorthand_scoped_let_does_not_replace_an_existing_root_binding(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let x = 1").ok
        assert s.eval_entry("let A::x = 2").ok

        root = s.eval_entry("x")
        scoped = s.eval_entry("A::x")

        assert root.ok, root.diagnostics
        assert scoped.ok, scoped.diagnostics
        assert root.value == IntValue(1)
        assert scoped.value == IntValue(2)

    def test_region_form_redeclaration_across_entries_replaces_the_retained_member(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nlet x = 1\nend A").ok

        replaced = s.eval_entry("scope A\nlet x = 99\nend A")
        result = s.eval_entry("A::x")

        assert replaced.ok, replaced.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(99)

    def test_cross_kind_replacement_from_binding_to_declaration(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok

        replaced = s.eval_entry("def A::x() -> int = 2")
        result = s.eval_entry("A::x()")

        assert replaced.ok, replaced.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_cross_kind_replacement_from_declaration_to_binding(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def A::x() -> int = 2").ok

        replaced = s.eval_entry("let A::x = 1")
        result = s.eval_entry("A::x")

        assert replaced.ok, replaced.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_member_cannot_be_reopened_as_a_nested_scope(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nlet B = 1\nend A").ok

        reopened = s.eval_entry("scope A::B\nlet x = 2\nend A::B")

        assert not reopened.ok
        original = s.eval_entry("A::B")
        assert original.ok, original.diagnostics
        assert original.value == IntValue(1)

    def test_same_named_bindings_at_different_paths_coexist_and_stay_distinct(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok
        assert s.eval_entry("let B::x = 2").ok

        a = s.eval_entry("A::x")
        b = s.eval_entry("B::x")

        assert a.ok, a.diagnostics
        assert b.ok, b.diagnostics
        assert a.value == IntValue(1)
        assert b.value == IntValue(2)

    def test_redeclaring_one_path_does_not_disturb_a_same_named_sibling_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok
        assert s.eval_entry("let B::x = 2").ok
        assert s.eval_entry("let A::x = 100").ok

        a = s.eval_entry("A::x")
        b = s.eval_entry("B::x")

        assert a.ok, a.diagnostics
        assert b.ok, b.diagnostics
        assert a.value == IntValue(100)
        assert b.value == IntValue(2)

    def test_retained_scoped_var_assignable_by_path_in_a_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nvar counter = 0\nend A").ok

        assign = s.eval_entry("A::counter := A::counter + 1")
        result = s.eval_entry("A::counter")

        assert assign.ok, assign.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_scoped_var_assignable_bare_after_use_in_a_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nvar counter = 0\nend A").ok
        assert s.eval_entry("A::counter := 5").ok

        assign = s.eval_entry("use A::*\ncounter := counter + 1")
        result = s.eval_entry("A::counter")

        assert assign.ok, assign.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(6)

    @pytest.mark.parametrize("binding", ["let", "var"])
    def test_use_selects_scoped_binding_declared_in_same_entry(self, binding: str) -> None:
        result = ReplSession().eval_entry(f"use A::{{x}}\n{binding} A::x = 1\nx")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_use_exposes_a_retained_scoped_type_alias(self) -> None:
        session = ReplSession()
        assert session.eval_entry("type A::Meters = int").ok
        assert session.eval_entry("use A::*").ok

        result = session.eval_entry("let distance: Meters = 1")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_use_exposes_a_retained_scoped_generic_type_alias(self) -> None:
        session = ReplSession()
        assert session.eval_entry("type A::Items[T] = array[T]").ok
        assert session.eval_entry("use A::*").ok

        result = session.eval_entry("let items: Items[int] = [1]")

        assert result.ok, result.diagnostics
        assert result.value == ArrayValue([IntValue(1)])

    def test_retained_relative_use_keeps_its_resolved_scope_target(self) -> None:
        session = ReplSession()
        assert session.eval_entry("scope Source\ndef value() -> int = 1\nend Source").ok
        assert session.eval_entry("scope Outer\nuse Source::*\nend Outer").ok
        assert session.eval_entry(
            "scope Outer\nscope Source\ndef value() -> int = 2\nend Source\nend Outer"
        ).ok

        declared = session.eval_entry("scope Outer\ndef selected() -> int = value()\nend Outer")
        result = session.eval_entry("Outer::selected()")

        assert declared.ok, declared.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_use_sees_a_member_promoted_by_a_later_entry(self) -> None:
        """A retained use resolves a member a later entry adds to its target."""
        s = ReplSession()
        assert s.eval_entry("scope A\nvar x = 1\nend A").ok
        assert s.eval_entry("use A::*").ok
        assert s.eval_entry("scope A\nvar y = 2\nend A").ok

        result = s.eval_entry("y")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_replacing_a_scoped_use_discards_its_empty_prior_region(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\ndef member() -> int = 1\nend A").ok
        assert s.eval_entry("scope B\nuse A::*\nend B").ok

        replacement = s.eval_entry("scope B\nuse A::*\nend B")

        assert replacement.ok, replacement.diagnostics

    def test_scoped_declarations_and_bindings_coexist_at_one_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok

        added = s.eval_entry("scope A\ndef doubled() -> int = x * 2\nend A")
        result = s.eval_entry("A::doubled()")

        assert added.ok, added.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_reset_clears_retained_scoped_bindings(self) -> None:
        s = ReplSession()
        assert s.eval_entry("let A::x = 1").ok

        s.reset()

        after_reset = s.eval_entry("A::x")
        redeclared = s.eval_entry("let A::x = 7")

        assert not after_reset.ok
        assert redeclared.ok, redeclared.diagnostics
        assert redeclared.value == IntValue(7)

    def test_echo_distinguishes_a_root_binding_from_a_scoped_one(self) -> None:
        s = ReplSession()

        root = s.eval_entry("let x = 10")
        scoped = s.eval_entry("let A::x = 20")

        assert root.ok, root.diagnostics
        assert scoped.ok, scoped.diagnostics
        assert root.name == "x"
        assert scoped.name == "A::x"


# ---------------------------------------------------------------------------
# Cross-entry scope/member collisions
# ---------------------------------------------------------------------------


class TestCrossEntryScopeCollision:
    """A member cannot claim a name owned by a retained nested scope layer.

    Mirrors the same-entry rule (a binding, a ``def``, a type, and a nested
    scope all collide at one path) across REPL entries: a name that a prior
    entry established as a nested scope's own path is not free for a later
    entry to claim as an ordinary member, for either the shorthand ``let`` or
    the shorthand ``def`` spelling.
    """

    def test_shorthand_let_cannot_claim_a_retained_nested_scopes_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nscope B\ndef q() -> int = 2\nend B\nend A").ok

        result = s.eval_entry("let A::B = 1")

        assert not result.ok
        still_reachable = s.eval_entry("A::B::q()")
        assert still_reachable.ok, still_reachable.diagnostics
        assert still_reachable.value == IntValue(2)

    def test_shorthand_def_cannot_claim_a_retained_nested_scopes_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nscope B\ndef q() -> int = 2\nend B\nend A").ok

        result = s.eval_entry("def A::B() -> int = 1")

        assert not result.ok
        still_reachable = s.eval_entry("A::B::q()")
        assert still_reachable.ok, still_reachable.diagnostics
        assert still_reachable.value == IntValue(2)

    def test_type_declared_at_a_retained_scopes_path_keeps_the_scopes_own_members(self) -> None:
        """A type declared at a path a scope previously occupied does not disturb it.

        Scope paths are namespaces, independent of whichever declaration (if
        any) occupies that path as a nominal type: a record declared at
        ``A::B`` shares the namespace position with a function ``A::B::q``
        declared there earlier, rather than colliding with or displacing it.
        """
        s = ReplSession()
        assert s.eval_entry("scope A\nscope B\ndef q() -> int = 2\nend B\nend A").ok
        assert s.eval_entry("record A::B()").ok

        result = s.eval_entry("A::B::q()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_unknown_member_of_a_retained_type_path_reports_a_diagnostic_not_a_crash(
        self,
    ) -> None:
        """A qualified reference into a retained type path for a name the type
        does not own must be a normal diagnostic, not an internal crash."""
        s = ReplSession()
        assert s.eval_entry("record A::B()").ok

        result = s.eval_entry("A::B::x")

        assert not result.ok
        assert any("not a member of 'A::B'" in d.message for d in result.diagnostics)


class TestBareConstructorVisibilityAcrossEntries:
    """A declaration's bare-name reach must not depend on the entry boundary.

    Whether a constructor spelling is usable unqualified is decided once, by
    where and how it was declared -- never by whether the reference happens
    to land in the same REPL entry as the declaration or a later one.
    """

    def test_record_in_a_named_scope_is_not_bare_across_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope S\nrecord Inner(v: int)\nend S").ok

        bare = s.eval_entry("Inner(v = 1)")
        qualified = s.eval_entry("S::Inner(v = 1)")

        assert not bare.ok
        assert qualified.ok, qualified.diagnostics

    def test_record_in_a_named_scope_is_not_bare_within_one_entry(self) -> None:
        s = ReplSession()

        result = s.eval_entry("scope S\nrecord Inner(v: int)\nend S\nInner(v = 1)")

        assert not result.ok

    def test_constructible_alias_in_a_named_scope_is_not_bare_across_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope S\nrecord Inner(v: int)\ntype Wrap = Inner\nend S").ok

        bare = s.eval_entry("Wrap(v = 1)")
        qualified = s.eval_entry("S::Wrap(v = 1)")

        assert not bare.ok
        assert qualified.ok, qualified.diagnostics

    def test_constructible_alias_in_a_named_scope_is_not_bare_within_one_entry(self) -> None:
        s = ReplSession()

        result = s.eval_entry(
            "scope S\nrecord Inner(v: int)\ntype Wrap = Inner\nend S\nWrap(v = 1)"
        )

        assert not result.ok

    def test_root_enum_inline_member_stays_bare_across_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum E\n  | Foo(v: int)").ok

        bare = s.eval_entry("Foo(v = 1)")

        assert bare.ok, bare.diagnostics
        assert isinstance(bare.value, RecordValue)
        assert bare.value.fields["v"] == IntValue(1)

    def test_root_enum_inline_member_stays_bare_within_one_entry(self) -> None:
        s = ReplSession()

        result = s.eval_entry("enum E\n  | Foo(v: int)\nFoo(v = 1)")

        assert result.ok, result.diagnostics
        assert isinstance(result.value, RecordValue)
        assert result.value.fields["v"] == IntValue(1)

    def test_root_enum_reference_to_a_scoped_record_stays_bare_across_entries(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope M\nrecord Go(amount: int)\nend M\nenum Step\n  | M::Go").ok

        bare = s.eval_entry("Go(amount = 1)")

        assert bare.ok, bare.diagnostics
        assert isinstance(bare.value, RecordValue)
        assert bare.value.fields["amount"] == IntValue(1)

    def test_root_enum_reference_to_a_scoped_record_stays_bare_within_one_entry(self) -> None:
        s = ReplSession()

        result = s.eval_entry(
            "scope M\nrecord Go(amount: int)\nend M\nenum Step\n  | M::Go\nGo(amount = 1)"
        )

        assert result.ok, result.diagnostics
        assert isinstance(result.value, RecordValue)
        assert result.value.fields["amount"] == IntValue(1)


# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------


class TestStdlib:
    def test_implicit_core_import_makes_names_available_unqualified(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        some_result = s.eval_entry("let present: Option[int] = Some(value = 1)")
        none_result = s.eval_entry("let missing: Option[int] = None")

        assert some_result.ok, some_result.diagnostics
        assert none_result.ok, none_result.diagnostics

    def test_type_of_uses_implicit_core_import(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        assert "Option::Some[int]" in s.type_of("Some(value = 1)")

    def test_retained_explicit_core_import_suppresses_later_preludes(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        assert s.eval_entry("import std/core").ok
        assert not s.eval_entry("Some(value = 1)").ok
        assert s.eval_entry("std/core::Option::Some(value = 1)").ok

    def test_no_stdlib_requires_explicit_core_import_after_reset(self) -> None:
        s = ReplSession(
            default_stdlib=False,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        )

        assert not s.eval_entry("Some(value = 1)").ok
        assert s.eval_entry("import std/core::*\nSome(value = 1)").ok

        s.reset()

        assert not s.eval_entry("Some(value = 1)").ok

    def test_core_stdlib_qualified_generic_type_resolves_in_type_definition(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        result = s.eval_entry("enum E = A(x: std/core::Option[int])")

        assert result.ok, result.diagnostics

    def test_prelude_record_name_echoes_as_constructor(self) -> None:
        s = ReplSession()

        result = s.eval_entry("ExecResult")

        assert result.ok, result.diagnostics
        assert result.kind == "expression"
        assert result.value is not None
        assert result.value_type is not None
        assert "ExecResult" in repr(result.value_type)

    def test_prelude_record_constructor_is_available(self) -> None:
        s = ReplSession()

        result = s.eval_entry(
            'ExecResult(stdout = "ok", exit_code = 0, stderr = "", timed_out = false)'
        )

        assert result.ok, result.diagnostics
        assert result.value is not None
        assert result.value_type is not None
        assert result.value_type.name == "ExecResult"

    def test_all_public_builtin_prelude_constructors_are_available(self) -> None:
        s = ReplSession()
        table = create_seeded_type_table()

        for name, typ in BUILTIN_PRELUDE_TYPES.items():
            if name in COMPATIBILITY_PRELUDE_TYPE_NAMES:
                continue
            typedef = BUILTIN_PRELUDE_TYPE_DEFS[name]
            if isinstance(typ, RecordType):
                result = s.eval_entry(f"{name}({_constructor_args(dict(typedef.fields))})")
                assert result.ok, (name, result.diagnostics)
                assert result.value_type is not None
                assert result.value_type.name == name
            elif isinstance(typ, EnumType):
                for member in typedef.members:
                    variant = member.name
                    args = _constructor_args(dict(table.record_fields(member)))
                    call = f"{name}::{variant}({args})" if args else f"{name}::{variant}"
                    result = s.eval_entry(call)
                    assert result.ok, (name, variant, result.diagnostics)
                    assert result.value_type is not None
                    assert result.value_type.name == variant

    def test_all_concrete_builtin_exceptions_are_available(self) -> None:
        s = ReplSession()
        table = create_seeded_type_table()

        for name, typ in BUILTIN_EXCEPTIONS.items():
            assert isinstance(typ, ExceptionType)
            if table.exception_def(typ).abstract:
                result = s.eval_entry(f'{name}(message = "x")')
                assert not result.ok
                assert any("abstract" in diagnostic.message for diagnostic in result.diagnostics)
                continue
            fields = table.exception_fields(typ)
            result = s.eval_entry(f"{name}({_constructor_args(fields)})")
            assert result.ok, (name, result.diagnostics)
            assert result.value_type is not None
            assert result.value_type.name == name


# ---------------------------------------------------------------------------
# Builtin identity across REPL entries
# ---------------------------------------------------------------------------

_EXEC_RESULT_FIELDS = "  stdout: text\n  exit_code: int\n  stderr: text\n  timed_out: bool\n"

_AGENT_VARIANTS = (
    "  | AgentCommand(command: text)\n"
    "  | AgentClaude(model: text, thinking: text)\n"
    "  | AgentCodex(model: text, thinking: text)\n"
    "  | AgentPi(provider: text, model: text, thinking: text)\n"
)

_AGENT_REQUEST_FIELDS = (
    "  agent: Agent\n"
    "  prompt: text\n"
    "  target_type: Option[text]\n"
    "  format_instructions: Option[text]\n"
    "  json_schema: Option[json]\n"
    "  attempt: int\n"
    "  previous_error: Option[text]\n"
    "  metadata: json\n"
)

_PARSE_POLICY_VARIANTS = "  | Abort\n  | Retry(n: int)\n"

# A plain (non-``builtin``) ``Option`` declaration, shaped like the standard
# library's own, for arrangements that declare their own host-contract types
# without loading the standard library: ``AgentRequest``'s canonical shape
# uses ``Option``-typed fields, and ``Option`` itself is an ordinary type the
# standard library happens to define -- never a ``builtin`` name of its own --
# so a program without the standard library must supply an equivalent one.
_OPTION_DECL = "enum Option[T] =\n  | None\n  | Some(value: T)\n"


def _session_with_import_root(root: Path) -> ReplSession:
    """Create a ``ReplSession`` with *root* as the only module search root."""
    from agm.agl.modules.roots import assemble_roots

    roots = assemble_roots(
        invocation_root=root,
        stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        lib_root=None,
        configured=[],
        cli=[],
        cwd=root,
    )
    s = ReplSession()
    s._roots = roots  # inject roots directly
    return s


class TestBuiltinIdentityAcrossEntries:
    """A REPL session's shared ``TypeTable`` accumulates ``builtin`` declarations
    across entries even though each entry re-checks only its own text. Both the
    link image that mints host values and the bare-name scan that types an
    unannotated ``exec()`` must track that accumulation instead of seeing only
    the current entry (which loses an earlier entry's declaration) or the
    first-ever entry (which keeps a stale one alive over a later redeclaration).
    """

    def test_host_minted_value_keeps_the_scoped_identity_declared_in_an_earlier_entry(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A value minted for a ``builtin`` type declared in one entry keeps that
        entry's own scoped identity when a LATER entry -- which re-checks only
        its own text -- mints a value of that type; it must not fall back to the
        shipped standard library's identity just because the later entry itself
        declares nothing."""
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            "scope Stdlib\n"
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def print[T](value: T) -> unit\n"
            "builtin def exec(command: text) -> ExecResult\n"
            "end Stdlib\n"
        )
        assert declare.ok, declare.diagnostics

        shell = FakeShell([{"command": "echo hey", "stdout": "hey"}])
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            call = s.eval_entry('Stdlib::print(Stdlib::exec("echo hey"))')
        shell.assert_complete()

        assert call.ok, call.diagnostics
        assert capsys.readouterr().out.strip() == (
            'Stdlib::ExecResult(stdout = "hey", exit_code = 0, stderr = "", timed_out = false)'
        )

    def test_unannotated_exec_types_as_the_most_recently_declared_builtin(self) -> None:
        """The bare-name scan behind a program's own ``ExecResult`` identity must
        resolve to the LIVE (most recently registered) declaration, not the
        oldest surviving one, when two entries each declare their own scoped
        ``builtin record ExecResult``."""
        s = ReplSession(default_stdlib=False)
        first = s.eval_entry(
            "scope A\n"
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def exec(command: text) -> ExecResult\n"
            "end A\n"
        )
        second = s.eval_entry(
            "scope B\n"
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def exec(command: text) -> ExecResult\n"
            "end B\n"
        )
        assert first.ok, first.diagnostics
        assert second.ok, second.diagnostics

        result = s.eval_entry('B::exec("echo hi")', check_only=True)

        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ("B",)

    def test_unpromoted_builtin_exception_declaration_does_not_type_a_later_catch_clause(
        self,
    ) -> None:
        """A ROOT ``builtin exception`` declaration the entry never promoted
        must not steer a later ``catch`` clause or the host's raise identity
        -- the exception-shaped counterpart of
        ``TestRedefinition.test_unpromoted_builtin_declaration_does_not_type_a_later_host_call``.
        Its own rollback path (``TypeEnvironment.restore_type_names_from``)
        must not skip a reserved name just because such a name is normally
        non-shadowable: it can appear among an entry's own unpromoted names
        only when that entry itself wrote the ``builtin`` declaration.

        Declared without the standard library and at ROOT: a root reserved
        name always conflicts with the standard library's own root
        declaration once loaded (see
        ``TestBuiltinIdentityWithStandardLibrary``), and a SCOPED name is
        never in ``restore_type_names_from``'s reserved-name set to begin
        with (it is keyed by the joined ``scope::name`` spelling, never the
        bare reserved one), so only a root declaration without the standard
        library actually exercises the skip this audit fixed.
        """
        s = ReplSession(default_stdlib=False)
        failed = s.eval_entry(
            'let stop: int = raise Abort(message = "stop")\n'
            "builtin exception RangeError extends Exception()"
        )
        assert not failed.ok
        # A RUNTIME (partial-promotion) failure, not a static rejection --
        # confirms this actually reached `restore_type_names_from` rather
        # than failing before any declaration could even be checked.
        assert failed.error is not None

        result = s.eval_entry(
            "let step = 0\n"
            "try\n"
            "  for i in 1 to 5 by step do\n"
            "    ()\n"
            "  done\n"
            "catch RangeError as error =>\n"
            "  ()"
        )
        assert result.ok, result.diagnostics
        assert result.error is None

    def test_unpromoted_builtin_record_declaration_does_not_displace_the_live_one(
        self,
    ) -> None:
        """An entry that redeclares a ROOT ``builtin record`` and then fails at
        runtime must leave the session's EARLIER declaration of that name in
        force: a later host call keeps minting the identity values already in
        the session carry, so comparing an old value against a fresh one is
        still a comparison of one type against itself.

        This is the rollback direction ``restore_type_names_from`` owns.
        Skipping a reserved bare name there instead leaves the unpromoted
        redeclaration holding the name, and the session then reports two
        identically-spelled ``ExecResult`` types as incomparable.
        """
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def exec(command: text) -> ExecResult\n"
            "builtin\nexception Exception\n  *\n  message: text\n"
            "builtin exception Abort extends Exception()\n"
        )
        assert declare.ok, declare.diagnostics
        original = s.eval_entry(
            'let a = ExecResult(stdout = "hi", exit_code = 0, stderr = "", timed_out = false)'
        )
        assert original.ok, original.diagnostics
        assert isinstance(original.value_type, RecordType)

        failed = s.eval_entry(
            'let stop: int = raise Abort(message = "stop")\n'
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
        )
        assert not failed.ok
        # A RUNTIME (partial-promotion) failure, so the redeclaration was
        # checked and then rolled back rather than rejected outright.
        assert failed.error is not None

        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            minted = s.eval_entry('let b = exec("echo hi")')
        assert minted.ok, minted.diagnostics
        assert isinstance(minted.value_type, RecordType)
        assert minted.value_type.decl_id == original.value_type.decl_id

        equal = s.eval_entry("a == b")
        assert equal.ok, equal.diagnostics
        assert equal.value == BoolValue(True)


# ---------------------------------------------------------------------------
# Builtin identity vs. the standard library's own canonical declarations
# ---------------------------------------------------------------------------


class TestBuiltinIdentityWithStandardLibrary:
    """A program's own ``builtin`` declaration of a reserved name must win the
    bare-name resolution the checker and the host both use, even though the
    standard library's own root declaration of the same name is ALSO
    registered in the shared ``TypeTable`` once the default standard library
    is loaded. Before the fix, the checker's and the host's independent
    resolutions disagreed whenever both were present: the checker picked the
    program's own declaration (a forward scan that sees it registered last)
    while the host picked whichever module's AST happened to be walked last
    -- silently falling back to the standard library's own identity even
    though the checker was typing calls against the program's own one.

    A ROOT ``builtin`` declaration of a reserved name is always rejected once
    the standard library is loaded (``validate_builtin_declaration_uniqueness``
    keys uniqueness on scope path + name alone, so a root declaration
    unconditionally collides with the standard library's own root one, in
    any module); every scenario below that needs a root declaration to
    succeed therefore runs without the standard library, and the "declared
    with the standard library" arrangement is instead covered by the
    dedicated rejection test, which applies identically to every ``builtin``
    kind since they all share that one uniqueness check.
    """

    def test_root_builtin_declaration_conflicts_with_the_standard_librarys_own(
        self,
    ) -> None:
        """A root ``builtin`` declaration of a reserved name always collides
        with the standard library's own root declaration of it -- a
        pre-existing invariant this fix leaves untouched, so "declared at
        root with the standard library loaded" is a rejection, not a
        success, for every ``builtin`` kind (record/enum/exception share one
        uniqueness namespace)."""
        s = ReplSession()
        declare = s.eval_entry(f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}")
        assert not declare.ok
        assert any("ExecResult" in d.message for d in declare.diagnostics)

    def test_scoped_builtin_record_declared_with_stdlib_types_and_mints_consistently(
        self,
    ) -> None:
        """The reported crash: with the scoped declaration losing the
        bare-name race, ``a.field`` raised an internal nominal-mismatch error
        and ``a == b`` was wrongly ``False`` even though ``a`` and ``b`` name
        the identical declaration."""
        s = ReplSession()
        declare = s.eval_entry(f"scope A\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end A\n")
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry('let a = exec("echo hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ("A",)

        ctor = s.eval_entry(
            'let b = A::ExecResult(stdout = "hi", exit_code = 0, stderr = "", timed_out = false)'
        )
        assert ctor.ok, ctor.diagnostics
        equal = s.eval_entry("let c = a == b")
        assert equal.ok, equal.diagnostics
        assert equal.value == BoolValue(True)

        field = s.eval_entry("a.exit_code")
        assert field.ok, field.diagnostics
        assert field.value == IntValue(0)

    def test_root_builtin_record_declared_without_stdlib_types_and_mints_consistently(
        self,
    ) -> None:
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def exec(command: text) -> ExecResult\n"
        )
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry('let a = exec("echo hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ()

        ctor = s.eval_entry(
            'let b = ExecResult(stdout = "hi", exit_code = 0, stderr = "", timed_out = false)'
        )
        assert ctor.ok, ctor.diagnostics
        equal = s.eval_entry("a == b")
        assert equal.ok, equal.diagnostics
        assert equal.value == BoolValue(True)

    def test_scoped_builtin_exception_declared_with_stdlib_is_caught_by_its_own_identity(
        self,
    ) -> None:
        """The ``builtin exception`` counterpart of the scoped-record repro
        above. ``catch`` names an exception type by ordinary LEXICAL name
        resolution (unlike ``exec``'s bare-name-anywhere default type), so
        the catching ``def`` is declared inside the same scope as the
        declaration -- exactly like the shipped standard library's own
        exception-catching code would be -- while the RAISED value's
        identity still comes from the host's scope-agnostic bare-name mint,
        which is what this fix keeps in agreement with it."""
        s = ReplSession()
        declare = s.eval_entry(
            "scope A\n"
            "builtin exception RangeError extends Exception()\n"
            "def trigger(step: int) -> unit =\n"
            "  try\n"
            "    for i in 1 to 5 by step do\n"
            "      ()\n"
            "    done\n"
            "  catch RangeError as error =>\n"
            "    ()\n"
            "end A\n"
        )
        assert declare.ok, declare.diagnostics

        result = s.eval_entry("A::trigger(0)")
        assert result.ok, result.diagnostics
        assert result.error is None

    def test_later_entry_builtin_declaration_does_not_retype_an_earlier_hosts_mint(
        self,
    ) -> None:
        """A value the host minted BEFORE a program declared its own
        ``builtin record`` keeps its own (canonical) identity; only a mint
        AFTER the declaration picks up the program's own one, and the two
        are unrelated nominal types -- the same supersession semantics an
        ordinary record redeclaration already has."""
        s = ReplSession()
        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            before = s.eval_entry('let a = exec("echo hi")')
        assert before.ok, before.diagnostics
        assert isinstance(before.value_type, RecordType)
        before_decl_id = before.value_type.decl_id

        declare = s.eval_entry(f"scope A\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end A\n")
        assert declare.ok, declare.diagnostics

        with patch("agm.core.process.run_capture_result", side_effect=shell):
            after = s.eval_entry('let b = exec("echo hi")')
        assert after.ok, after.diagnostics
        assert isinstance(after.value_type, RecordType)
        assert after.value_type.decl_id != before_decl_id
        assert after.value_type.scope_path == ("A",)

        still_reads_old_field = s.eval_entry("a.exit_code")
        cross = s.eval_entry("a == b")
        assert still_reads_old_field.ok, still_reads_old_field.diagnostics
        assert still_reads_old_field.value == IntValue(0)
        assert not cross.ok  # unrelated nominal types: a static error, not False


# ---------------------------------------------------------------------------
# Builtin identity across imported library modules
# ---------------------------------------------------------------------------


class TestBuiltinIdentityAcrossModules:
    """A ``builtin`` declaration inside an imported library module must type
    and mint through the exact same shared ``TypeTable`` resolution as one
    written directly in the entry: the host and the checker both read
    whichever declaration the table's bare-name resolution currently
    answers with, regardless of which module wrote it. Every declaration
    below is scoped: an unscoped one would collide with the standard
    library's own root declaration (see
    ``TestBuiltinIdentityWithStandardLibrary``), and two unscoped
    declarations from different modules would collide with EACH OTHER --
    ``validate_builtin_declaration_uniqueness`` keys uniqueness on scope path
    + name only, ignoring which module a declaration came from.
    """

    def _make_session_with_root(self, root: Path) -> ReplSession:
        """Create a ReplSession with *root* as the only module search root."""
        return _session_with_import_root(root)

    def test_builtin_declared_in_an_imported_library_module_types_and_mints_consistently(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            f"scope Lib\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end Lib\n"
        )
        s = self._make_session_with_root(tmp_path)
        declare = s.eval_entry("import lib")
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry('let a = exec("echo hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert isinstance(result.value, RecordValue)
        assert result.value.nominal.value == result.value_type.decl_id

        ctor = s.eval_entry(
            'let b = lib::Lib::ExecResult(stdout = "hi", exit_code = 0, stderr = "", '
            "timed_out = false)"
        )
        assert ctor.ok, ctor.diagnostics
        equal = s.eval_entry("a == b")
        assert equal.ok, equal.diagnostics
        assert equal.value == BoolValue(True)

    def test_two_modules_declaring_the_same_bare_builtin_name_still_agree_on_identity(
        self, tmp_path: Path
    ) -> None:
        """Two imported modules, each with their own scoped ``builtin record
        ExecResult`` (at different scope paths, so neither collides with the
        other), leave the shared ``TypeTable``'s bare-name tie-break to pick
        a winner (last-registered, unaffected by this fix -- see
        ``TypeTable.builtin_declaration``); what this fix guarantees is that
        the checker and the host agree on whichever declaration that is, not
        which declaration wins."""
        (tmp_path / "lib_a.agl").write_text(
            f"scope X\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end X\n"
        )
        (tmp_path / "lib_b.agl").write_text(
            f"scope Y\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end Y\n"
        )
        s = self._make_session_with_root(tmp_path)
        declare = s.eval_entry("import lib_a\nimport lib_b")
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="hi")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry('let a = exec("echo hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert isinstance(result.value, RecordValue)
        assert result.value.nominal.value == result.value_type.decl_id


# ---------------------------------------------------------------------------
# Builtin identity for the other host-contract nominals: ``AgentRequest``
# (``ask-request``'s result), ``Agent`` (the ``agent`` argument to
# ``ask``/``ask-request``), and ``ParsePolicy`` (``on_parse_error``).
#
# Every one of these resolutions goes through a program's own ``builtin``
# declaration of the name (``BuiltinCallChecker._builtin_contract_type``), so
# the identity a host call is typed against is always the identity the host
# actually mints. The classes below are the direct counterparts of
# ``TestBuiltinIdentity*`` above, covering the same arrangements
# (root/scoped, with/without the standard library, same-entry/earlier-entry/
# imported-module declarations, and the no-declaration-at-all regression)
# for each of the three.
# ---------------------------------------------------------------------------


class TestAgentRequestBuiltinIdentity:
    """``ask-request``'s result type (``BuiltinCallChecker.check_ask_request``)."""

    def test_scoped_agent_request_declared_with_stdlib_types_and_mints_consistently(self) -> None:
        """The reported crash: with the scoped declaration losing the
        bare-name race, ``q.prompt`` raised an internal nominal-mismatch
        error even though ``q`` was minted against that very declaration."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}end A\n"
        )
        assert declare.ok, declare.diagnostics

        result = s.eval_entry('let q = ask-request("hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ("A",)

        field = s.eval_entry("q.prompt")
        assert field.ok, field.diagnostics
        assert field.value == TextValue("hi")

    def test_agent_request_declared_in_the_same_entry_as_the_ask_request_call(self) -> None:
        s = ReplSession()
        result = s.eval_entry(
            f"scope A\nbuiltin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}end A\n"
            'let q = ask-request("hi")'
        )
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ("A",)

    def test_scoped_agent_and_agent_request_declared_together_with_agent_omitted_rejected(
        self,
    ) -> None:
        """Both ``Agent`` and ``AgentRequest`` declared together at the same
        scope, with the ``agent`` argument OMITTED entirely: the host still
        fills the field from the canonical default agent regardless, into a
        field statically typed as this scope's own ``Agent`` -- the contract
        is incoherent whether or not a value is explicitly supplied for the
        argument that field holds."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}"
            f"builtin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}end A\n"
        )
        assert declare.ok, declare.diagnostics

        result = s.eval_entry('let q = ask-request("hi")')
        assert not result.ok
        assert any("AgentRequest" in d.message and "agent" in d.message for d in result.diagnostics)

    def test_root_agent_request_declared_without_stdlib_rejected_as_incoherent(self) -> None:
        """Without the standard library, a root ``AgentRequest`` whose own
        ``agent`` field types to this program's own root ``Agent`` (the only
        ``Agent`` there is here, since nothing seeds a canonical one without
        the standard library) is an incoherent contract: the host always
        fills that field with the standard ``Agent`` identity, never
        whatever declaration the contract's own field type happens to name,
        so the call is rejected rather than minting a value whose identity
        disagrees with its static field type."""
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            f"{_OPTION_DECL}"
            f"builtin\nenum Agent\n{_AGENT_VARIANTS}"
            f"builtin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}"
            'builtin def ask-request(prompt: text, agent: Agent = AgentCommand(command = "noop")) '
            "-> AgentRequest\n"
        )
        assert declare.ok, declare.diagnostics

        # No standard library, so ``std/config::default-agent`` -- the
        # ``agent`` parameter's canonical default -- was never declared;
        # supplying ``agent`` explicitly is unrelated to the fix under test.
        result = s.eval_entry('let q = ask-request("hi", agent = AgentCommand(command = "noop"))')
        assert not result.ok
        assert any("AgentRequest" in d.message and "agent" in d.message for d in result.diagnostics)

    def test_ask_request_without_named_program_syntax_types_as_canonical_agent_request(
        self,
    ) -> None:
        """Regression: a program that declares none of its own builtin types
        keeps ``ask-request``'s canonical (root) ``AgentRequest`` identity
        exactly as before this fix."""
        s = ReplSession()
        result = s.eval_entry('let q = ask-request("hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ()

        field = s.eval_entry("q.prompt")
        assert field.ok, field.diagnostics
        assert field.value == TextValue("hi")

    def test_agent_request_declared_in_an_imported_library_module_types_and_mints_consistently(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            f"scope Lib\nbuiltin record AgentRequest\n{_AGENT_REQUEST_FIELDS}end Lib\n"
        )
        s = _session_with_import_root(tmp_path)
        declare = s.eval_entry("import lib")
        assert declare.ok, declare.diagnostics

        result = s.eval_entry('let q = ask-request("hi")')
        assert result.ok, result.diagnostics
        assert isinstance(result.value_type, RecordType)
        assert result.value_type.scope_path == ("Lib",)

        field = s.eval_entry("q.prompt")
        assert field.ok, field.diagnostics
        assert field.value == TextValue("hi")


class TestAgentArgumentBuiltinIdentity:
    """The ``agent`` argument to ``ask``/``ask-request``
    (``BuiltinCallChecker._validate_ask_like_arguments``)."""

    def test_scoped_agent_value_rejected_as_ask_request_agent_argument(self) -> None:
        """A value of the program's own scoped ``Agent`` is rejected as the
        ``agent`` argument to ``ask-request``.

        Only ``Agent`` is redeclared here (matching the reported repro
        exactly), not ``AgentRequest``, so ``AgentRequest``'s own ``agent``
        field keeps its canonical (root) static field type: the value's
        differently-scoped ``Agent`` is an ordinary static type mismatch
        against it, restoring the clean, pre-existing diagnostic instead of
        the internal crash that accepting the mismatched value used to lead
        to at evaluation.
        """
        s = ReplSession()
        declare = s.eval_entry(f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}end A\n")
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g = A::Agent::AgentCommand("echo")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('let q = ask-request("hi", agent = g)')
        assert not result.ok
        assert any("A::Agent" in d.message for d in result.diagnostics)

    def test_scoped_agent_and_agent_request_declared_together_rejected_as_incoherent(
        self,
    ) -> None:
        """A program that redeclares BOTH ``Agent`` and ``AgentRequest`` at
        the same scope gets an incoherent contract: ``AgentRequest.agent``
        resolves to that same scoped ``Agent``, not the standard identity the
        host actually fills the field with, so the call is rejected rather
        than minting a field whose static type disagrees with its value."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}"
            f"builtin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}end A\n"
        )
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g = A::Agent::AgentCommand("echo")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('let q = ask-request("hi", agent = g)')
        assert not result.ok
        assert any("AgentRequest" in d.message and "agent" in d.message for d in result.diagnostics)

    def test_scoped_agent_value_rejected_as_ask_agent_argument(self) -> None:
        """``ask`` shares ``_validate_ask_like_arguments`` with ``ask-request``,
        so it rejects the same scoped ``Agent`` value the same way; checked
        only (an actual agent dispatch is out of scope here)."""
        s = ReplSession()
        declare = s.eval_entry(f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}end A\n")
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g = A::Agent::AgentCommand("echo")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('ask("hi", agent = g)', check_only=True)
        assert not result.ok
        assert any("A::Agent" in d.message for d in result.diagnostics)

    def test_root_agent_value_without_stdlib_rejected_as_ask_request_agent_argument(self) -> None:
        """Without the standard library, a root ``AgentRequest`` whose own
        ``agent`` field types to this program's own root ``Agent`` is
        incoherent regardless of which value is supplied for ``agent``: the
        contract itself is rejected before its argument is even checked."""
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            f"{_OPTION_DECL}"
            f"builtin\nenum Agent\n{_AGENT_VARIANTS}"
            f"builtin\nrecord AgentRequest\n{_AGENT_REQUEST_FIELDS}"
            'builtin def ask-request(prompt: text, agent: Agent = AgentCommand(command = "noop")) '
            "-> AgentRequest\n"
        )
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g = AgentClaude("sonnet", "medium")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('let q = ask-request("hi", agent = g)')
        assert not result.ok
        assert any("AgentRequest" in d.message and "agent" in d.message for d in result.diagnostics)

    def test_unrelated_value_is_still_rejected_as_the_agent_argument(self) -> None:
        """Regression: passing a value of an unrelated type as ``agent`` is
        still a static rejection -- the shape-mismatch direction of this fix,
        confirmed with the program's own scoped ``Agent`` also live."""
        s = ReplSession()
        declare = s.eval_entry(f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}end A\n")
        assert declare.ok, declare.diagnostics

        not_agent = s.eval_entry("enum NotAgent\n  | X")
        assert not_agent.ok, not_agent.diagnostics

        result = s.eval_entry('ask-request("hi", agent = NotAgent::X)', check_only=True)
        assert not result.ok
        assert any("NotAgent" in d.message for d in result.diagnostics)

    def test_agent_declared_in_an_imported_library_module_rejected_as_agent_argument(
        self, tmp_path: Path
    ) -> None:
        """Only ``Agent`` is declared by the library module here (not
        ``AgentRequest``), so -- as in
        ``test_scoped_agent_value_rejected_as_ask_request_agent_argument``
        above -- the value's own (differently-scoped) ``Agent`` is an
        ordinary static type mismatch against ``AgentRequest``'s canonical
        field type."""
        (tmp_path / "lib.agl").write_text(
            f"scope Lib\nbuiltin enum Agent\n{_AGENT_VARIANTS}end Lib\n"
        )
        s = _session_with_import_root(tmp_path)
        declare = s.eval_entry("import lib")
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g = lib::Lib::Agent::AgentCommand("echo")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('let q = ask-request("hi", agent = g)')
        assert not result.ok
        assert any("Lib::Agent" in d.message for d in result.diagnostics)

    def test_scoped_agent_receiver_rejected_as_ask_request_receiver(self) -> None:
        """The receiver of ``x.ask-request(...)`` IS the agent the host stores
        in ``AgentRequest.agent``, so it is held to the same identity
        requirement as the ``agent`` named argument. Reaching evaluation with
        a differently-scoped ``Agent`` receiver instead mints a request whose
        ``agent`` field value disagrees with its static type, which an
        exhaustive ``case q.agent of`` then cannot dispatch."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}"
            "builtin def Agent::ask-request(self, prompt: text) -> AgentRequest\n"
            "end A\n"
        )
        assert declare.ok, declare.diagnostics

        g = s.eval_entry('let g: A::Agent = A::Agent::AgentCommand("echo")')
        assert g.ok, g.diagnostics

        result = s.eval_entry('let q = g.ask-request("hi")')
        assert not result.ok
        assert any("A::Agent" in d.message for d in result.diagnostics)

    def test_canonical_agent_receiver_still_builds_a_request(self) -> None:
        """Regression: the ordinary receiver form still works and still mints
        a request whose ``agent`` field is readable."""
        s = ReplSession()
        result = s.eval_entry(
            'let agent: Agent = AgentCommand("echo")\nlet q = agent.ask-request("hi")'
        )
        assert result.ok, result.diagnostics
        field = s.eval_entry("q.prompt")
        assert field.ok, field.diagnostics
        assert field.value == TextValue("hi")


class TestHostRaisedExceptionContractIdentity:
    """A ``builtin exception`` the host raises carries the same requirement a
    host-minted record does: the host fills its nominal-typed fields with
    standard identities, so a program's own redeclaration of one of those
    names cannot be what such a field resolves to."""

    _AGENT_CALL_ERROR = (
        "builtin\nexception AgentCallError extends Exception\n"
        "  *\n  agent: Agent\n  cause: text\n  metadata: json\n"
    )

    def test_scoped_agent_and_agent_call_error_declared_together_rejected(self) -> None:
        """``AgentCallError.agent`` resolving to a sibling scoped ``Agent``
        makes the caught value's ``agent`` field statically that scoped enum
        while the host always raises with the standard one -- rejected at the
        ``catch`` clause rather than left to fail dispatching a ``case`` over
        the field at evaluation."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum Agent\n{_AGENT_VARIANTS}{self._AGENT_CALL_ERROR}"
            "def trigger() -> text =\n"
            "  try\n"
            '    ask("hi")\n'
            "  catch AgentCallError as e =>\n"
            "    case e.agent of\n"
            "      | AgentCommand(command) => command\n"
            "      | AgentClaude(model, thinking) => model\n"
            "      | AgentCodex(model, thinking) => model\n"
            "      | AgentPi(provider, model, thinking) => model\n"
            "end A\n"
        )
        assert not declare.ok
        assert any(
            "AgentCallError" in d.message and "agent" in d.message for d in declare.diagnostics
        )

    def test_scoped_agent_call_error_over_the_standard_agent_is_still_caught(self) -> None:
        """Regression: with nothing shadowing ``Agent``, the same scoped
        ``builtin exception AgentCallError`` keeps the standard identity in
        its own ``agent`` field, so it is caught by its own declaration and
        its field is dispatched over successfully."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\n{self._AGENT_CALL_ERROR}"
            "def trigger() -> text =\n"
            "  try\n"
            '    ask("hi")\n'
            "  catch AgentCallError as e =>\n"
            "    case e.agent of\n"
            "      | AgentCommand(command) => command\n"
            "      | AgentClaude(model, thinking) => model\n"
            "      | AgentCodex(model, thinking) => model\n"
            "      | AgentPi(provider, model, thinking) => model\n"
            "end A\n"
        )
        assert declare.ok, declare.diagnostics

        result = s.eval_entry("A::trigger()")
        assert result.ok, result.diagnostics
        assert result.error is None
        assert isinstance(result.value, TextValue)


class TestParsePolicyBuiltinIdentity:
    """``on_parse_error``'s static ``ParsePolicy`` constructor recognition
    (``BuiltinCallChecker._extract_parse_policy_str`` /
    ``_accepts_as_parse_policy_constructor``)."""

    def test_scoped_parse_policy_constructor_accepted_by_exec(self) -> None:
        """The reported rejection: a static constructor of the program's own
        scoped ``ParsePolicy``, written at its own qualified path
        (``A::ParsePolicy::Retry``), was rejected because the qualifier
        check only ever accepted the bare root spelling."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum ParsePolicy =\n{_PARSE_POLICY_VARIANTS}end A\n"
        )
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="2")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry(
                'let r = exec::[int]("echo hi", on_parse_error = A::ParsePolicy::Retry(n = 2))'
            )
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_scoped_parse_policy_abort_constructor_accepted_by_exec(self) -> None:
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum ParsePolicy =\n{_PARSE_POLICY_VARIANTS}end A\n"
        )
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="2")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry(
                'let r = exec::[int]("echo hi", on_parse_error = A::ParsePolicy::Abort)'
            )
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_root_parse_policy_without_stdlib_accepted_by_exec(self) -> None:
        s = ReplSession(default_stdlib=False)
        declare = s.eval_entry(
            f"builtin\nrecord ExecResult\n{_EXEC_RESULT_FIELDS}"
            f"builtin\nenum ParsePolicy =\n{_PARSE_POLICY_VARIANTS}"
            "builtin def exec(command: text) -> ExecResult\n"
        )
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="2")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry('let r = exec::[int]("echo hi", on_parse_error = Retry(n = 2))')
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)

    def test_on_parse_error_without_named_program_syntax_still_accepts_canonical_forms(
        self,
    ) -> None:
        """Regression: a program that declares none of its own builtin types
        keeps every canonical ``on_parse_error`` spelling accepted exactly as
        before this fix."""
        s = ReplSession()
        shell = FakeShell(stdout="2")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            bare = s.eval_entry('let a = exec::[int]("echo hi", on_parse_error = Retry(n = 2))')
        assert bare.ok, bare.diagnostics
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            qualified = s.eval_entry(
                'let b = exec::[int]("echo hi", on_parse_error = ParsePolicy::Abort)'
            )
        assert qualified.ok, qualified.diagnostics

    def test_on_parse_error_rejects_a_local_binding_shadowing_abort(self) -> None:
        """The reported bug: a local ``let Abort = ...`` binding shadows the
        ``ParsePolicy::Abort`` constructor's bare spelling, so the checker
        must resolve ``on_parse_error``'s value through real name
        resolution rather than matching the raw spelling ``Abort`` -- a
        shadowing local binding is not a constructor at all and is rejected
        exactly like any other non-constructor expression there."""
        s = ReplSession()
        result = s.eval_entry(
            "let Abort = ParsePolicy::Retry(n = 3)\n"
            'let n: int = exec::[int]("echo 7", on_parse_error = Abort)\nn',
            check_only=True,
        )
        assert not result.ok
        assert any("on_parse_error" in d.message for d in result.diagnostics)

    def test_on_parse_error_rejects_a_local_binding_shadowing_abort_call_form(self) -> None:
        """The call-form (``Abort()``) counterpart of the shadowing bug:
        it bypassed real resolution the same way the bare-spelling form
        did, and is rejected the same way."""
        s = ReplSession()
        result = s.eval_entry(
            "let Abort = ParsePolicy::Retry(n = 3)\n"
            'let n: int = exec::[int]("echo 7", on_parse_error = Abort())\nn',
            check_only=True,
        )
        assert not result.ok
        assert any("on_parse_error" in d.message for d in result.diagnostics)

    def test_on_parse_error_rejects_a_local_binding_shadowing_retry(self) -> None:
        """The ``Retry`` counterpart: a local binding shadowing ``Retry``'s
        bare spelling is rejected rather than silently reinterpreted as a
        ``ParsePolicy::Retry`` call spelled the same way."""
        s = ReplSession()
        result = s.eval_entry(
            'let Retry = 5\nlet n: int = exec::[int]("echo 7", on_parse_error = Retry(n = 3))\nn',
            check_only=True,
        )
        assert not result.ok
        assert any("on_parse_error" in d.message for d in result.diagnostics)

    def test_on_parse_error_rejects_a_qualifier_naming_an_unrelated_enum(self) -> None:
        """Regression: an unrelated enum's constructor is still rejected as
        ``on_parse_error``, with the program's own scoped ``ParsePolicy``
        also live -- the shape-mismatch direction of this fix."""
        s = ReplSession()
        declare = s.eval_entry(
            f"scope A\nbuiltin\nenum ParsePolicy =\n{_PARSE_POLICY_VARIANTS}end A\n"
        )
        assert declare.ok, declare.diagnostics
        not_policy = s.eval_entry("enum NotPolicy\n  | Abort")
        assert not_policy.ok, not_policy.diagnostics

        result = s.eval_entry(
            'let n: int = exec::[int]("ls", on_parse_error = NotPolicy::Abort())', check_only=True
        )
        assert not result.ok
        assert any("ParsePolicy" in d.message for d in result.diagnostics)

    def test_parse_policy_declared_in_an_imported_library_module_accepted_by_exec(
        self, tmp_path: Path
    ) -> None:
        """A wildcard import brings ``Lib::ParsePolicy`` into scope at its own
        path without the module route prefix -- the qualifier spelling this
        fix recognizes (a module-route-qualified spelling like
        ``lib::Lib::ParsePolicy::Retry`` is outside this fix's scope, exactly
        as it was for the canonical ``ParsePolicy`` before it: an on_parse_error
        constructor was never recognized through an import route prefix)."""
        (tmp_path / "lib.agl").write_text(
            f"scope Lib\nbuiltin enum ParsePolicy =\n{_PARSE_POLICY_VARIANTS}end Lib\n"
        )
        s = _session_with_import_root(tmp_path)
        declare = s.eval_entry("import lib::*")
        assert declare.ok, declare.diagnostics

        shell = FakeShell(stdout="2")
        with patch("agm.core.process.run_capture_result", side_effect=shell):
            result = s.eval_entry(
                'let r = exec::[int]("echo hi", on_parse_error = Lib::ParsePolicy::Retry(n = 2))'
            )
        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)


# ---------------------------------------------------------------------------
# Redefinition / shadowing
# ---------------------------------------------------------------------------


class TestRedefinition:
    def test_let_redefined_with_new_type_shadows(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        r = s.eval_entry('let x = "hello"')
        assert r.ok
        assert isinstance(r.value_type, TextType)
        # The promoted binding now has the new type/value.
        promoted = {n: (t, v) for n, t, v in s.bindings()}
        typ, _val = promoted["x"]
        assert isinstance(typ, TextType)

    def test_record_redefinition_shadows(self) -> None:
        s = ReplSession()
        s.eval_entry("record R\n  a: int")
        r = s.eval_entry("record R\n  b: text")
        assert r.ok
        assert r.kind == "declaration"
        assert r.name == "R"
        # The new shape is the one in effect.
        use = s.eval_entry('let r = R(b = "hi")')
        assert use.ok
        bad = s.eval_entry("let r2 = R(a = 1)")
        assert not bad.ok  # old field 'a' no longer valid

    def test_builtin_agent_redeclaration_preserves_prior_values(self) -> None:
        declaration = """\
builtin
enum Agent
  | AgentCommand(command: text)
  | AgentClaude(model: text, thinking: text)
  | AgentCodex(model: text, thinking: text)
  | AgentPi(provider: text, model: text, thinking: text)
"""
        session = ReplSession(default_stdlib=False)
        assert session.eval_entry(declaration).ok
        assert session.eval_entry('let stale = AgentClaude("sonnet", "medium")').ok
        assert session.eval_entry(declaration).ok

        stale = session.eval_entry("stale")
        fresh = session.eval_entry('AgentCommand(command = "echo hello")')

        assert stale.ok, stale.diagnostics
        assert isinstance(stale.value_type, RecordType)
        assert stale.value_type.name == "AgentClaude"
        assert fresh.ok, fresh.diagnostics
        assert isinstance(fresh.value_type, RecordType)
        assert fresh.value_type.name == "AgentCommand"

    def test_referenced_enum_keeps_its_original_record_member_after_record_supersession(
        self,
    ) -> None:
        """A retained enum selects its member by handle, not its record's current name."""
        session = ReplSession()
        assert session.eval_entry("record R(old: int)").ok
        assert session.eval_entry("type OldR = R").ok
        assert session.eval_entry("enum E = ::R").ok
        assert session.eval_entry("let old: E = R(old = 1)").ok
        assert session.eval_entry("record R(fresh: text)").ok

        matched = session.eval_entry("case old of\n  | R(old) => old")
        tested = session.eval_entry("old is R")
        narrowed = session.eval_entry("(old as OldR).old")
        not_a_member = session.eval_entry('let fresh: E = R(fresh = "new")')

        assert matched.ok, matched.diagnostics
        assert matched.value == IntValue(1)
        assert tested.ok, tested.diagnostics
        assert tested.value == BoolValue(True)
        assert narrowed.ok, narrowed.diagnostics
        assert narrowed.value == IntValue(1)
        assert not not_a_member.ok

    def test_referenced_generic_member_preserves_its_applied_field_for_json_casts(self) -> None:
        """A referenced member applies the enum's arguments to its own fields."""
        session = ReplSession()
        assert session.eval_entry("record Box[A](value: A)").ok
        assert session.eval_entry("enum Result[T] = ::Box[T]").ok
        assert session.eval_entry("let box = Box(value = 1)").ok
        assert session.eval_entry("let result: Result[int] = box").ok

        encoded = session.eval_entry("let wire = result as json")
        decoded = session.eval_entry("wire as Result[int]")

        assert encoded.ok, encoded.diagnostics
        assert decoded.ok, decoded.diagnostics
        assert isinstance(decoded.value, RecordValue)
        assert decoded.value.fields == {"value": IntValue(1)}

    def test_enum_supersession_remints_inline_members_without_invalidating_old_ones(self) -> None:
        """Old and new enum-member handles remain independently matchable and castable."""
        session = ReplSession()
        assert session.eval_entry("enum E\n  | A(old: int)").ok
        assert session.eval_entry("type OldA = E::A").ok
        assert session.eval_entry("let old: E = A(old = 1)").ok
        assert session.eval_entry("enum E\n  | B(fresh: text)").ok
        assert session.eval_entry('let new: E = B(fresh = "new")').ok

        old_case = session.eval_entry("case old of\n  | A(old) => old")
        old_is = session.eval_entry("old is A")
        old_cast = session.eval_entry("(old as OldA).old")
        new_case = session.eval_entry("case new of\n  | B(fresh) => fresh")
        new_is = session.eval_entry("new is B")
        new_cast = session.eval_entry("(new as E::B).fresh")
        current_member_on_old_value = session.eval_entry("old as E::B")

        assert old_case.ok, old_case.diagnostics
        assert old_case.value == IntValue(1)
        assert old_is.ok, old_is.diagnostics
        assert old_is.value == BoolValue(True)
        assert old_cast.ok, old_cast.diagnostics
        assert old_cast.value == IntValue(1)
        assert new_case.ok, new_case.diagnostics
        assert new_case.value == TextValue("new")
        assert new_is.ok, new_is.diagnostics
        assert new_is.value == BoolValue(True)
        assert new_cast.ok, new_cast.diagnostics
        assert new_cast.value == TextValue("new")
        assert not current_member_on_old_value.ok

    def test_same_spelling_enum_member_supersession_keeps_old_and_new_identities_incompatible(
        self,
    ) -> None:
        """A reused enum/member spelling never bridges its old and new identities."""
        session = ReplSession()
        assert session.eval_entry("enum E\n  | A(old: int)").ok
        assert session.eval_entry("type OldE = E").ok
        assert session.eval_entry("type OldA = E::A").ok
        assert session.eval_entry("let old: OldE = E::A(old = 1)").ok
        assert session.eval_entry("enum E\n  | A(fresh: text)").ok
        assert session.eval_entry('let new: E = E::A(fresh = "new")').ok

        old_to_new_cast = session.eval_entry("old as E::A")
        new_to_old_cast = session.eval_entry("new as OldA")
        old_to_new_assignment = session.eval_entry("let current: E = old")
        new_to_old_assignment = session.eval_entry("let previous: OldE = new")
        old_case_against_new_member = session.eval_entry("case old of\n  | E::A(fresh) => fresh")
        new_case_against_old_member = session.eval_entry("case new of\n  | OldA(old) => old")

        assert not old_to_new_cast.ok
        assert not new_to_old_cast.ok
        assert not old_to_new_assignment.ok
        assert not new_to_old_assignment.ok
        assert not old_case_against_new_member.ok
        assert not new_case_against_old_member.ok

    def test_record_redefinition_clears_generic_metadata(self) -> None:
        s = ReplSession()
        first = s.eval_entry("record Box[T]\n  x: T")
        assert first.ok
        second = s.eval_entry("record Box\n  x: int")
        assert second.ok

        use = s.eval_entry("let y: Box[int] = Box(x = 1)")

        assert not use.ok
        assert any("does not take type arguments" in d.message for d in use.diagnostics)

    def test_redeclaring_a_generic_record_as_non_generic_gives_the_name_to_the_new_one(
        self,
    ) -> None:
        """A bare construction after the redeclaration builds the NEW record.

        The construction itself must work rather than crash, and the value it
        builds must belong to the newest declaration: comparing it with a
        value of the superseded generic declaration is a type error, because
        the two declarations are unrelated types.
        """
        s = ReplSession()
        assert s.eval_entry("record Box[T]\n  x: T").ok
        assert s.eval_entry("let old = Box(x = 1)").ok
        assert s.eval_entry("record Box\n  x: int").ok

        result = s.eval_entry("let fresh = Box(x = 1)")
        cross = s.eval_entry("old == fresh")

        assert result.ok, result.diagnostics
        assert isinstance(result.value, RecordValue)
        assert result.value.display_name == "Box"
        assert result.value.fields == {"x": IntValue(1)}
        assert not cross.ok

    def test_redeclaring_a_record_as_an_enum_reports_a_diagnostic_not_a_crash(self) -> None:
        """A record's construction spelling does not outlive its declaration.

        Once the name belongs to an enum, the record form it used to accept
        is reported as a diagnostic rather than routed into the enum's
        variant-constructor path, which has no variant to build.
        """
        s = ReplSession()
        assert s.eval_entry("record R(a: int)").ok
        assert s.eval_entry("enum R\n  | V(b: int)").ok

        stale = s.eval_entry("R(a = 3)")
        fresh = s.eval_entry("R::V(b = 4)")

        assert not stale.ok
        assert fresh.ok, fresh.diagnostics

    def test_redeclaring_an_enum_as_a_record_drops_stale_variants(self) -> None:
        session = ReplSession()
        assert session.eval_entry("enum Color | Red").ok
        assert session.eval_entry("record Color(value: int)").ok

        stale_use = session.eval_entry("use Color::{Red}")
        fresh = session.eval_entry("Color(value = 1)")

        assert not stale_use.ok
        assert fresh.ok, fresh.diagnostics

    def test_redeclaring_an_enum_preserves_unrelated_retained_static_members(self) -> None:
        session = ReplSession()
        assert session.eval_entry("enum Color | Red").ok
        assert session.eval_entry("def Color::code() -> int = 42").ok

        redeclared = session.eval_entry("enum Color | Blue\nColor::code()")
        retained = session.eval_entry("Color::code()")
        obsolete = session.eval_entry("Color::Red")

        assert redeclared.ok, redeclared.diagnostics
        assert redeclared.value == IntValue(42)
        assert retained.ok, retained.diagnostics
        assert retained.value == IntValue(42)
        assert not obsolete.ok

    def test_redeclaring_a_type_preserves_nested_constructor_members(self) -> None:
        session = ReplSession()
        assert session.eval_entry("enum Color | Old").ok
        assert session.eval_entry("record Color::Meta(value: int)").ok
        assert session.eval_entry("use Color::*").ok
        assert session.eval_entry("enum Color | New").ok

        nested = session.eval_entry("Meta(value = 1)")

        assert nested.ok, nested.diagnostics
        assert isinstance(nested.value, RecordValue)
        assert nested.value.display_name == "Color::Meta"

    def test_redeclaring_an_enum_retires_types_nested_under_an_old_member(self) -> None:
        session = ReplSession()
        assert session.eval_entry("enum Color | Old").ok
        assert session.eval_entry("record Color::Old::Meta(value: int)").ok

        assert session.eval_entry("enum Color | New").ok

        retired = session.eval_entry("Color::Old::Meta(value = 1)")
        fresh = session.eval_entry("Color::New")
        assert not retired.ok
        assert fresh.ok, fresh.diagnostics

    def test_redeclaring_a_used_enum_drops_its_stale_bare_variant(self) -> None:
        """A local use recorded before the enum is redeclared must not
        resurrect a variant the redeclaration's fresh member layer dropped."""
        s = ReplSession()
        assert s.eval_entry("enum Color\n  | Red\n  | Green").ok
        assert s.eval_entry("use Color::*").ok
        assert s.eval_entry("enum Color\n  | Blue").ok

        stale = s.eval_entry("Red")
        fresh = s.eval_entry("Blue")

        assert not stale.ok
        assert fresh.ok, fresh.diagnostics

    def test_failed_record_redefinition_restores_previous_methods(self) -> None:
        """A failed entry that would have redeclared a type changes nothing.

        The previous declaration, its methods, and a binding built against it
        all remain in effect exactly as before the failed entry.
        """
        session = ReplSession()
        assert session.eval_entry("record R(value: int)").ok
        assert session.eval_entry("def R::get(self) -> int = self.value").ok
        assert session.eval_entry("let existing = R(value = 7)").ok

        failed = session.eval_entry(
            'let stop: int = raise Abort(message = "stop")\nrecord R(value: text)'
        )

        assert not failed.ok
        result = session.eval_entry("R(value = 42).get()")
        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)
        existing_method = session.eval_entry("existing.get()")
        assert existing_method.ok, existing_method.diagnostics
        assert existing_method.value == IntValue(7)

    def test_unpromoted_redeclaration_leaves_the_previous_declaration_owning_the_name(
        self,
    ) -> None:
        """A redeclaration the entry never promoted does not keep the name.

        The previous declaration owns the name afterwards, so a later method
        declaration is checked against ITS members: a method colliding with a
        field it has is rejected, and one named after a field only the
        rolled-back declaration had is accepted and callable.
        """
        session = ReplSession()
        assert session.eval_entry("record R(a: int)").ok

        failed = session.eval_entry(
            'let stop: int = raise Abort(message = "stop")\nrecord R(b: int)'
        )

        assert not failed.ok
        collides = session.eval_entry("def R::a(self) -> int = 1")
        allowed = session.eval_entry("def R::b(self) -> int = 2")
        call = session.eval_entry("R(a = 1).b()")
        assert not collides.ok
        assert allowed.ok, allowed.diagnostics
        assert call.ok, call.diagnostics
        assert call.value == IntValue(2)

    def test_unpromoted_exception_is_not_a_member_conflict_for_a_later_method(self) -> None:
        """A never-promoted exception cannot make a later method collide.

        A retained SUPERSEDED declaration still owns its own members, because
        values built from it survive; a declaration the entry never promoted
        has no values and no name, so a method named after one of its fields
        must be accepted rather than rejected against a type the session
        cannot even name.
        """
        session = ReplSession()
        assert session.eval_entry("exception Base extends Exception\n  a: int").ok

        failed = session.eval_entry(
            'let stop: int = raise Abort(message = "stop")\nexception Ghost extends Base\n  b: int'
        )
        assert not failed.ok
        assert not session.eval_entry('Ghost(a = 1, b = 2, message = "x")').ok

        method = session.eval_entry("def Base::b(self) -> int = self.a")

        assert method.ok, method.diagnostics
        call = session.eval_entry('Base(a = 7, message = "m").b()')
        assert call.ok, call.diagnostics
        assert call.value == IntValue(7)

    def test_unpromoted_builtin_declaration_does_not_type_a_later_host_call(self) -> None:
        """A never-promoted ``builtin`` declaration must not steer a later entry.

        The bare-name scan that types an unannotated ``exec()`` reads the
        newest registered ``builtin ExecResult`` declaration; a declaration
        the entry never promoted is not one, so the call keeps the type the
        surviving declaration gives it — the same one the host actually mints.
        """
        session = ReplSession(default_stdlib=False)
        declared = session.eval_entry(
            f"builtin record ExecResult\n{_EXEC_RESULT_FIELDS}"
            "builtin def exec(command: text) -> ExecResult\n"
        )
        assert declared.ok, declared.diagnostics

        failed = session.eval_entry(
            'let stop: int = raise Abort(message = "stop")\n'
            f"scope Ghost\nbuiltin record ExecResult\n{_EXEC_RESULT_FIELDS}end Ghost"
        )
        assert not failed.ok

        call = session.eval_entry('exec("echo hi")', check_only=True)

        assert call.ok, call.diagnostics
        assert isinstance(call.value_type, RecordType)
        assert call.value_type.scope_path == ()


# ---------------------------------------------------------------------------
# Recursive types across entries
# ---------------------------------------------------------------------------


class TestRecursiveTypesAcrossEntries:
    """A recursive type declared in one entry is usable in later entries.

    The REPL (program context) re-validates the accumulated table on every entry;
    this is idempotent for a declaration that already passed inhabitation in
    the entry that declared it.
    """

    def test_recursive_enum_constructed_and_matched_in_later_entries(self) -> None:
        s = ReplSession()
        declare = s.eval_entry("enum Tree\n  | Leaf\n  | Node(value: int, left: Tree, right: Tree)")
        assert declare.ok

        build = s.eval_entry(
            "let t: Tree = Node(value = 1, left = Leaf(), right = Node(value = 2, left = Leaf(), "
            "right = Leaf()))"
        )
        assert build.ok

        match = s.eval_entry("case t of\n  | Leaf() => 0\n  | Node(value, left, right) => value")
        assert match.ok
        assert match.value is not None

    def test_recursive_record_still_rejects_redefinition_style_reset(self) -> None:
        # Redefinition semantics are unaffected: a later entry redeclaring
        # the same name with a different (still recursive) shape shadows it,
        # exactly like any other record redefinition.
        s = ReplSession()
        first = s.eval_entry("record Category\n  name: text\n  subcategories: array[Category]")
        assert first.ok
        second = s.eval_entry("record Category\n  label: text\n  kids: array[Category]")
        assert second.ok
        use = s.eval_entry('let c = Category(label = "root", kids = [])')
        assert use.ok
        stale = s.eval_entry('let bad = Category(name = "root", subcategories = [])')
        assert not stale.ok

    def test_binding_before_redeclaration_keeps_reading_old_fields_and_rendering(self) -> None:
        """A record redeclaration is supersession, not invalidation.

        A binding built against the old declaration keeps reading its own
        (old) fields and rendering the same way, while a new construction
        under the same name uses the new shape; accessing a field the old
        declaration never had still fails, exactly as it always would.
        """
        s = ReplSession()
        assert s.eval_entry("record R\n  old: int").ok
        assert s.eval_entry("let stale = R(old = 1)").ok
        old_render = s.eval_entry("stale")
        assert old_render.ok
        assert s.eval_entry("record R\n  fresh: int").ok

        old_field = s.eval_entry("stale.old")
        old_render_after = s.eval_entry("stale")
        fresh = s.eval_entry("R(fresh = 2)")
        missing_field = s.eval_entry("stale.fresh")

        assert old_field.ok, old_field.diagnostics
        assert old_field.value == IntValue(1)
        assert old_render_after.ok
        assert old_render_after.value == old_render.value
        assert fresh.ok, fresh.diagnostics
        assert not missing_field.ok

    def test_old_declarations_methods_still_resolve_new_declaration_starts_with_none(
        self,
    ) -> None:
        s = ReplSession()
        assert s.eval_entry("record R(value: int)").ok
        assert s.eval_entry("def R::get(self) -> int = self.value").ok
        assert s.eval_entry("let old = R(value = 1)").ok
        assert s.eval_entry("record R(value: text)").ok

        old_method = s.eval_entry("old.get()")
        new_method = s.eval_entry('R(value = "x").get()')

        assert old_method.ok, old_method.diagnostics
        assert old_method.value == IntValue(1)
        assert not new_method.ok

    def test_old_and_new_typed_values_are_never_comparable(self) -> None:
        """Two values of the same (old) declaration compare equal; across
        declarations, equality is a static type error even when both share
        one display name — the two are unrelated nominal types."""
        s = ReplSession()
        assert s.eval_entry("record R(value: int)").ok
        assert s.eval_entry("let a = R(value = 1)").ok
        assert s.eval_entry("let b = R(value = 1)").ok
        assert s.eval_entry("record R(value: int)").ok
        assert s.eval_entry("let c = R(value = 1)").ok

        same_old = s.eval_entry("a == b")
        cross = s.eval_entry("a == c")

        assert same_old.ok, same_old.diagnostics
        assert same_old.value == BoolValue(True)
        assert not cross.ok

    def test_catch_clause_matches_the_declaration_in_scope_where_it_is_written(self) -> None:
        s = ReplSession()
        assert s.eval_entry("exception E extends Exception()").ok
        assert s.eval_entry('let old_exc = E(message = "old")').ok
        assert s.eval_entry(
            'def catch_only_old() -> text = try raise old_exc catch E as e => "caught-old"'
        ).ok
        assert s.eval_entry("exception E extends Exception()").ok
        assert s.eval_entry('let new_exc = E(message = "new")').ok

        # A ``catch E`` written after the redeclaration binds the new E: it
        # catches a freshly raised new-E value but not the retained old one.
        catches_new = s.eval_entry('try raise new_exc catch E as e => "caught-new-clause"')
        does_not_catch_old = s.eval_entry('try raise old_exc catch E as e => "caught-new-clause"')
        # A ``catch E`` compiled before the redeclaration keeps matching only
        # the old E, unaffected by the redeclaration that came afterward.
        still_catches_old = s.eval_entry("catch_only_old()")

        assert catches_new.ok, catches_new.diagnostics
        assert catches_new.value == TextValue("caught-new-clause")
        assert not does_not_catch_old.ok
        assert does_not_catch_old.error is not None
        assert does_not_catch_old.error.type_name == "E"
        assert still_catches_old.ok, still_catches_old.diagnostics
        assert still_catches_old.value == TextValue("caught-old")

    def test_enum_variant_on_an_old_typed_value_survives_redeclaration(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum Color\n  | Red(shade: int)\n  | Green").ok
        assert s.eval_entry("let old: Color = Color::Red(shade = 1)").ok
        assert s.eval_entry("enum Color\n  | Blue").ok

        old_match = s.eval_entry("case old of\n  | Red(shade) => shade\n  | Green() => 0")
        fresh = s.eval_entry("Color::Blue")
        cross_match = s.eval_entry("case old of\n  | Blue() => 1\n  | Green() => 0")

        assert old_match.ok, old_match.diagnostics
        assert old_match.value == IntValue(1)
        assert fresh.ok, fresh.diagnostics
        assert not cross_match.ok

    def test_phantom_generic_member_on_a_retained_value_survives_enum_redeclaration(self) -> None:
        """A fieldless member remains matchable without recovering its enum arguments."""
        s = ReplSession()
        assert s.eval_entry("enum E[T]\n  | A").ok
        assert s.eval_entry("let old = A").ok
        assert s.eval_entry("enum E[T]\n  | B").ok

        matched = s.eval_entry("case old of\n  | A() => 1")

        assert matched.ok, matched.diagnostics
        assert matched.value == IntValue(1)

    def test_superseded_enum_members_do_not_suggest_the_reused_enum_annotation(self) -> None:
        """An old enum's members cannot be joined through its reused name."""
        s = ReplSession()
        assert s.eval_entry("enum Choice\n  | Yes\n  | No").ok

        current = s.eval_entry("[Choice::Yes, Choice::No]", check_only=True)

        assert not current.ok
        assert any(
            "annotate the literal with its enum type" in diagnostic.message.lower()
            for diagnostic in current.diagnostics
        )

        assert s.eval_entry("let yes = Choice::Yes\nlet no = Choice::No").ok
        assert s.eval_entry("enum Choice\n  | Maybe").ok

        mismatch = s.eval_entry("[yes, no]", check_only=True)

        assert not mismatch.ok
        assert all(
            "annotate the literal with its enum type" not in diagnostic.message.lower()
            for diagnostic in mismatch.diagnostics
        )

    def test_type_qualified_variant_pattern_names_the_newest_enum_declaration(self) -> None:
        """A qualifier is a type name, so it names the newest declaration.

        A bare constructor pattern follows the subject's own declaration and
        keeps destructuring a value built before the redeclaration, but a
        pattern that spells the enum out qualifies against the enum the name
        means now — which is not the subject's — and is rejected.
        """
        s = ReplSession()
        assert s.eval_entry("enum E\n  | A(x: int)").ok
        assert s.eval_entry("let old: E = E::A(x = 1)").ok
        assert s.eval_entry("enum E\n  | A(x: int)").ok

        bare = s.eval_entry("case old of\n  | A(x) => x")
        qualified = s.eval_entry("case old of\n  | E::A(x) => x")
        fresh = s.eval_entry("let fresh: E = E::A(x = 2)\ncase fresh of\n  | E::A(x) => x")

        assert bare.ok, bare.diagnostics
        assert bare.value == IntValue(1)
        assert not qualified.ok
        assert fresh.ok, fresh.diagnostics
        assert fresh.value == IntValue(2)

    def test_scoped_binding_before_redeclaration_keeps_using_its_old_declaration(self) -> None:
        """A scoped record redeclaration supersedes rather than invalidates.

        A scoped function returning a scoped record built before the record
        is redeclared keeps returning a value of the OLD declaration —
        readable through its own (old) field — after the redeclaration.
        """
        s = ReplSession()
        assert s.eval_entry("scope A\nrecord R(n: int)\nend A").ok
        assert s.eval_entry("scope A\ndef make() -> A::R = A::R(1)\nend A").ok
        call = s.eval_entry("A::make().n")
        assert call.ok
        assert call.value == IntValue(1)
        assert s.eval_entry("scope A\nrecord R(m: text)\nend A").ok

        still_old = s.eval_entry("A::make().n")
        missing_new_field = s.eval_entry("A::make().m")
        fresh = s.eval_entry('let fresh: A::R = A::R("hi")')

        assert still_old.ok, still_old.diagnostics
        assert still_old.value == IntValue(1)
        assert not missing_new_field.ok
        assert fresh.ok, fresh.diagnostics

    def test_ask_with_recursive_output_type_does_not_crash(self) -> None:
        """The REPL's contract-preview path (make_contract) handles a recursive ask target.

        Regression coverage for the graph-session contract-preview call
        (``materialize_contract`` → ``JsonCodec.make_contract`` →
        ``derive_schema``/``build_decode_schema``) with a recursive type: before
        the decode side supported ``$defs``/``RefDecode``, ``build_decode_schema``
        recursed forever on a finite recursive type and crashed lowering.
        """
        agent = CountingAgent(
            '{"$case": "Node", "value": 1, "left": {"$case": "Leaf"}, "right": {"$case": "Leaf"}}'
        )
        s = ReplSession(agent_dispatcher=agent)
        declare = s.eval_entry("enum Tree\n  | Leaf\n  | Node(value: int, left: Tree, right: Tree)")
        assert declare.ok
        asked = s.eval_entry('let t: Tree = ask """build a tree"""')
        assert asked.ok
        assert agent.calls == 1
        match = s.eval_entry("case t of\n  | Leaf() => 0\n  | Node(value, left, right) => value")
        assert match.ok
        assert match.value == IntValue(1)


# ---------------------------------------------------------------------------
# Echo data
# ---------------------------------------------------------------------------


class TestEchoData:
    def test_expression_echo_value_type_kind(self) -> None:
        s = ReplSession()
        r = s.eval_entry("3 * 4")
        assert r.kind == "expression"
        assert r.name is None
        assert r.value is not None and _int(r.value) == 12
        assert isinstance(r.value_type, IntType)

    @pytest.mark.parametrize(
        ("source", "rendered"),
        (
            ('Agent::AgentCommand("runner")', 'Agent::AgentCommand(\n  command = "runner"\n)'),
            (
                'Agent::AgentClaude("sonnet", "medium")',
                'Agent::AgentClaude(\n  model = "sonnet",\n  thinking = "medium"\n)',
            ),
            (
                'Agent::AgentCodex("o3", "high")',
                'Agent::AgentCodex(\n  model = "o3",\n  thinking = "high"\n)',
            ),
            (
                'Agent::AgentPi("openai", "gpt", "low")',
                'Agent::AgentPi(\n  provider = "openai",\n  model = "gpt",\n  thinking = "low"\n)',
            ),
        ),
    )
    def test_qualified_builtin_agent_constructor_echoes_its_surface_form(
        self, source: str, rendered: str
    ) -> None:
        """REPL echoes every qualified built-in Agent constructor unambiguously."""
        from agm.agl.repl.render import render_entry_result

        result = ReplSession().eval_entry(source)

        assert result.ok, result.diagnostics
        assert render_entry_result(result, echo=True) == rendered

    @pytest.mark.parametrize("binder", ("let", "var"))
    def test_trailing_binder_echoes_declared_value(self, binder: str) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        r = s.eval_entry(f"{binder} total = 5")

        assert r.kind == "binding"
        assert r.name == "total"
        assert r.value == IntValue(5)
        assert isinstance(r.value_type, IntType)
        assert render_entry_result(r, echo=True) == "total : int = 5"
        assert _int({name: value for name, _typ, value in s.bindings()}["total"]) == 5

    @pytest.mark.parametrize("binder", ("let", "var"))
    def test_shorthand_scoped_binder_echoes_its_full_path_name(self, binder: str) -> None:
        """A scoped binder's echo name distinguishes it from a same-named root binding.

        Regression: naming previously reused the bare binder name for a
        scoped ``let``/``var``, so ``A::total`` and a root ``total`` were
        indistinguishable in the echo.
        """
        s = ReplSession()
        r = s.eval_entry(f"{binder} A::total = 5")

        assert r.kind == "binding"
        assert r.name == "A::total"
        assert r.value == IntValue(5)

    @pytest.mark.parametrize("binder", ("let", "var"))
    def test_trailing_discard_binder_evaluates_and_echoes_nothing(self, binder: str) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        r = s.eval_entry(f"{binder} _ = 6")

        assert r.ok, r.diagnostics
        assert r.kind == "statement"
        assert render_entry_result(r, echo=True) is None
        assert s.bindings() == []

    def test_multi_item_entry_keeps_normal_block_discard_strictness(self) -> None:
        result = ReplSession().eval_entry("let first = 1\nfirst\nlet second = 2")

        assert not result.ok

    def test_trailing_pattern_let_echoes_whole_value_and_promotes_every_binder(self) -> None:
        from agm.agl.repl.render import render_entry_result

        session = ReplSession()
        assert session.eval_entry("record Pair\n  left: int\n  right: int").ok

        live = session.eval_entry("let Pair(left, right) = Pair(left = 2, right = 3)")
        checked = session.eval_entry(
            "let Pair(left, right) = Pair(left = 2, right = 3)", check_only=True
        )

        assert live.ok, live.diagnostics
        assert live.kind == "binding"
        assert live.name is None
        assert strip_decl_ids(live.value_type) == RecordType("Pair")
        assert render_entry_result(live, echo=True) == ": Pair = Pair(\n  left = 2,\n  right = 3\n)"
        assert checked.ok, checked.diagnostics
        assert checked.kind == "binding"
        assert checked.name is None
        assert strip_decl_ids(checked.value_type) == RecordType("Pair")
        assert (
            render_entry_result(checked, echo=True, check_only=True)
            == ": record Pair\n  left: int\n  right: int"
        )
        assert {name: value for name, _typ, value in session.bindings()} == {
            "left": IntValue(2),
            "right": IntValue(3),
        }
        assert session.eval_entry("left + right").value == IntValue(5)

    def test_constructor_pattern_without_binders_echoes_but_discard_does_not(self) -> None:
        from agm.agl.repl.render import render_entry_result

        session = ReplSession()
        assert session.eval_entry("record Pair\n  left: int\n  right: int").ok

        constructor = session.eval_entry("let Pair() = Pair(left = 2, right = 3)")
        discard = session.eval_entry("let _ = Pair(left = 4, right = 5)")

        assert constructor.ok, constructor.diagnostics
        assert constructor.kind == "binding"
        assert constructor.name is None
        assert strip_decl_ids(constructor.value_type) == RecordType("Pair")
        assert (
            render_entry_result(constructor, echo=True)
            == ": Pair = Pair(\n  left = 2,\n  right = 3\n)"
        )
        assert discard.ok, discard.diagnostics
        assert discard.kind == "statement"
        assert render_entry_result(discard, echo=True) is None
        assert session.bindings() == []

    def test_declaration_echo_kind(self) -> None:
        s = ReplSession()
        r = s.eval_entry("type Age = int")
        assert r.kind == "declaration"
        assert r.name == "Age"
        assert r.value is None

    def test_assign_stmt_echo_kind(self) -> None:
        # In AgL, ``:=`` is the only binder-kind that maps to "statement"
        # (it mutates an existing binding, has no new name, yields unit).
        s = ReplSession()
        s.eval_entry("var v = 0")
        r = s.eval_entry("v := 1")
        assert r.kind == "statement"
        assert r.value is None
        assert r.ok

    def test_print_call_echo_kind(self) -> None:
        # ``print`` is a function call, but it yields void so REPL echo suppresses it.
        s = ReplSession()
        r = s.eval_entry("print 1")
        assert r.kind == "expression"
        assert r.ok
        assert r.value == VOID_VALUE
        assert isinstance(r.value, UnitValue)
        assert not r.value.printable_in_repl

    def test_unit_literal_echoes_printable_unit(self) -> None:
        s = ReplSession()
        r = s.eval_entry("()")
        assert r.kind == "expression"
        assert r.ok
        assert isinstance(r.value, UnitValue)
        assert r.value.printable_in_repl

    def test_loop_echo_value_is_void(self) -> None:
        s = ReplSession()
        r = s.eval_entry("do[0] () done")
        assert r.kind == "expression"
        assert r.ok
        assert r.value == VOID_VALUE
        assert isinstance(r.value, UnitValue)
        assert not r.value.printable_in_repl


# ---------------------------------------------------------------------------
# type_of — type without evaluation, no state change, no agent
# ---------------------------------------------------------------------------


class TestTypeOf:
    def test_type_of_returns_canonical_type(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        assert s.type_of("x + 1") == repr(IntType())

    def test_type_of_displays_record_fields(self) -> None:
        s = ReplSession()
        s.eval_entry("record Point\n  x: int\n  y: text")
        s.eval_entry('let p = Point(x = 1, y = "north")')

        assert s.type_of("p") == "record Point\n  x: int\n  y: text"

    def test_type_of_scoped_nominal_displays_its_path(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope Left\nrecord Token()\nend Left\nlet token = Left::Token()").ok

        assert s.type_of("token") == "record Left::Token()"

    def test_type_of_displays_enum_constructors(self) -> None:
        s = ReplSession()
        s.eval_entry("enum Result\n  | Ok(value: int)\n  | Err(message: text)\n  | Unknown")
        s.eval_entry("let r = Ok(value = 1)")

        assert s.type_of("r") == "record Result::Ok\n  value: int"

    def test_type_of_resolves_prior_entry_constructor(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum Result\n  | Ok(value: int)\n  | Err(message: text)").ok

        expected = "record Result::Ok\n  value: int"
        assert s.type_of("Ok(value = 1)") == expected
        assert s.type_of("Result::Ok(value = 1)") == expected

    def test_type_of_does_not_promote_or_advance(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        before = [(n, repr(t)) for n, t, _v in s.bindings()]
        source_before = s.dump_source()
        s.type_of("x * 99")
        after = [(n, repr(t)) for n, t, _v in s.bindings()]
        assert before == after
        assert s.dump_source() == source_before
        # A subsequent real binding still works (node ids not corrupted).
        r = s.eval_entry("let y = x")
        assert r.ok

    def test_type_of_fires_no_agent(self) -> None:
        agent = CountingAgent("RESULT")
        s = ReplSession(agent_dispatcher=agent)
        # type_of an agent-calling expression must NOT dispatch.
        assert s.type_of('ask """ask"""') == repr(TextType())
        assert agent.calls == 0

    def test_type_of_rejects_non_expression(self) -> None:
        s = ReplSession()
        with pytest.raises(AglError):
            s.type_of("let q = 1")

    def test_type_of_propagates_type_error(self) -> None:
        from agm.agl.typecheck import AglTypeError

        s = ReplSession()
        s.eval_entry('let s = "x"')
        with pytest.raises(AglTypeError):
            s.type_of("s + 1")

    def test_type_of_propagates_match_compilation_error(self) -> None:
        s = ReplSession()
        with pytest.raises(AglError, match="Non-exhaustive"):
            s.type_of("case true of | true => 1")


# ---------------------------------------------------------------------------
# Inference boundaries
# ---------------------------------------------------------------------------


class TestInferenceBoundaries:
    def test_generic_entry_infers_concrete_result_in_graph_and_type_sessions(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        assert s.eval_entry("def id[T](x: T) -> T = x").ok
        assert s.eval_entry("def app[T](f: T -> T, x: T) -> T = f(x)").ok

        program_result = s.eval_entry("let result = app(id, 0)")

        assert program_result.ok, program_result.diagnostics
        assert program_result.value == IntValue(0)
        assert isinstance(program_result.value_type, IntType)
        assert render_entry_result(program_result, echo=True) == "result : int = 0"
        # ``eval_entry`` and ``:type`` both use the program pipeline against
        # the same persisted declarations.
        assert s.type_of("app(id, 0)") == "int"

    def test_failed_generic_entries_leave_next_entry_fresh_and_render_diagnostic_notes(
        self,
    ) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        assert s.eval_entry("def id[T](x: T) -> T = x").ok
        assert s.eval_entry("def app[T](f: T -> T, x: T) -> T = f(x)").ok
        assert s.eval_entry("def same[T](left: T, right: T) -> T = left").ok

        unresolved = s.eval_entry("id")
        assert not unresolved.ok
        conflicting = s.eval_entry('same(0, "x")')
        assert not conflicting.ok
        assert conflicting.diagnostics[0].related
        rendered = render_entry_result(conflicting, echo=True)
        assert rendered is not None
        assert "note:" in rendered

        # Both failures are entry-local: the succeeding occurrence receives a
        # new generic instantiation rather than any stale solver constraints.
        recovered = s.eval_entry('app(id, "fresh")')
        assert recovered.ok, recovered.diagnostics
        assert recovered.value_type == TextType()
        assert s.eval_entry("app(id, 7)").value_type == IntType()

    def test_failed_entry_does_not_persist_a_generic_declaration(self) -> None:
        s = ReplSession()

        failed = s.eval_entry("def transient[T](x: T) -> T = x\ntransient")

        assert not failed.ok
        assert not s.eval_entry("transient(1)").ok
        # A subsequent declaration/use starts from only committed state.
        assert s.eval_entry("def id[T](x: T) -> T = x").ok
        assert s.eval_entry("id(1)").value_type == IntType()


# ---------------------------------------------------------------------------
# Atomic-on-error
# ---------------------------------------------------------------------------


class TestFailureEffects:
    def test_type_error_leaves_bindings_unchanged(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 10")
        before = _snapshot(s)
        r = s.eval_entry('let b = a + "oops"')
        assert not r.ok
        assert r.diagnostics
        assert r.error is None
        assert _snapshot(s) == before

    def test_runtime_raise_preserves_completed_binding(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 10")
        r = s.eval_entry("let before = 20\nlet z: decimal = 1 / 0")
        assert not r.ok
        assert r.error is not None  # mapped RunError, not a pre-exec diagnostic
        assert r.diagnostics == []
        assert r.installed == ("before",)
        use = s.eval_entry("before + a")
        assert use.ok
        assert use.value is not None and _int(use.value) == 30

    def test_runtime_raise_promotes_complete_pattern_let_but_not_later_bindings(self) -> None:
        session = ReplSession()
        assert session.eval_entry("record Pair\n  left: int\n  right: int").ok

        failed = session.eval_entry(
            "let Pair(left, right) = Pair(left = 2, right = 3)\n"
            'let later: int = raise Abort(message = "stop")'
        )

        assert not failed.ok
        assert failed.installed == ("left", "right")
        assert {name: value for name, _typ, value in session.bindings()} == {
            "left": IntValue(2),
            "right": IntValue(3),
        }
        assert session.eval_entry("left + right").value == IntValue(5)
        assert not session.eval_entry("later").ok

    def test_installed_report_includes_promoted_type_declaration(self) -> None:
        # Regression: a promoted RECORD/ENUM/EXCEPTION/type-alias declaration is
        # tracked separately (``promoted_type_names``) from the value bindings
        # that build ``installed``, so it used to be silently dropped from the
        # "Installed before failure" report even though it fully survived and a
        # later entry could use it.
        session = ReplSession()

        failed = session.eval_entry("record Point\n  x: int\nlet z = [1, 2][9]")

        assert not failed.ok
        assert "Point" in failed.installed
        assert session.eval_entry("Point(x = 1)").ok

    def test_completed_pattern_initializer_and_function_metadata_survive_called_failure(
        self,
    ) -> None:
        session = ReplSession()
        assert session.eval_entry("record Pair\n  left: int\n  right: int").ok

        failed = session.eval_entry(
            "let Pair(left, right) = Pair(left = 2, right = 3)\n"
            'def fail() -> int = raise Abort(message = "stop")\n'
            "fail()"
        )

        assert not failed.ok
        assert {"left", "right", "fail"} <= set(failed.installed)
        assert session.eval_entry("left + right").value == IntValue(5)
        assert session.type_of("fail") == "() -> int"
        assert session.eval_entry("fail()").error is not None

    def test_runtime_failure_excludes_function_with_uninitialized_value_dependency(self) -> None:
        session = ReplSession()

        failed = session.eval_entry(
            'def needs_value() -> int = value\nlet value: int = raise Abort(message = "stop")'
        )

        assert not failed.ok
        assert "needs_value" not in failed.installed
        assert not session.eval_entry("needs_value()").ok

    def test_runtime_failure_excludes_function_with_unpromoted_nominal_dependency(self) -> None:
        session = ReplSession()

        failed = session.eval_entry(
            "type Delayed = Later\n"
            "def needs_type(value: dict[text, Delayed]) -> dict[text, Delayed] = value\n"
            'let stop: int = raise Abort(message = "stop")\n'
            "record Later\n"
            "  value: int"
        )

        assert not failed.ok
        assert "needs_type" not in failed.installed
        assert "Delayed" not in session.type_names()
        assert not session.eval_entry("needs_type").ok
        assert not session.eval_entry("Later(value = 1)").ok

    def test_runtime_failure_promotes_generic_function_independent_of_same_named_nominal(
        self,
    ) -> None:
        session = ReplSession()

        failed = session.eval_entry(
            "def identity[T](value: T) -> T = value\n"
            'let stop: int = raise Abort(message = "stop")\n'
            "record T\n"
            "  value: int"
        )

        assert not failed.ok
        assert "identity" in failed.installed
        assert session.eval_entry("identity(1)").value == IntValue(1)
        assert not session.eval_entry("T(value = 1)").ok

    def test_runtime_failure_distinguishes_same_named_scoped_nominal_dependencies(
        self,
    ) -> None:
        session = ReplSession()

        failed = session.eval_entry(
            "scope A\n"
            "record T(value: int)\n"
            "end A\n"
            "def keep(value: A::T) -> A::T = value\n"
            'let stop: int = raise Abort(message = "stop")\n'
            "scope B\n"
            "record T(value: int)\n"
            "end B"
        )

        assert not failed.ok
        assert "keep" in failed.installed
        kept = session.eval_entry("keep(A::T(value = 1))")
        assert kept.ok, kept.diagnostics
        assert not session.eval_entry("B::T(value = 1)").ok

    def test_runtime_failure_retains_independent_completed_function_and_pattern_binders(
        self,
    ) -> None:
        session = ReplSession()
        assert session.eval_entry("record Pair\n  left: int\n  right: int").ok

        failed = session.eval_entry(
            "let Pair(left, right) = Pair(left = 2, right = 3)\n"
            "def sum() -> int = left + right\n"
            'def fail() -> int = raise Abort(message = "stop")\n'
            'let stop: int = raise Abort(message = "stop")'
        )

        assert not failed.ok
        assert {"left", "right", "sum", "fail"} <= set(failed.installed)
        assert session.eval_entry("sum()").value == IntValue(5)
        assert session.eval_entry("fail()").error is not None

    def test_runtime_raise_in_else_branch_returns_entry_error(self) -> None:
        s = ReplSession()

        result = s.eval_entry('let x = if 0 == 1 => 1 else => raise Abort(message = "a")')

        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert not s.eval_entry("x").ok
        assert s.eval_entry("1").ok

    def test_runtime_raise_does_not_install_failing_binding_from_prior_function(self) -> None:
        s = ReplSession()
        declare = s.eval_entry('def f[T]() -> T = raise Abort(message = "A")')
        assert declare.ok

        result = s.eval_entry("let x: int = f()")

        assert not result.ok
        assert result.error is not None
        assert result.installed == ()
        assert [name for name, _typ, _value in s.bindings()] == ["f"]
        assert not s.eval_entry("x").ok

    def test_runtime_raise_preserves_completed_param(self) -> None:
        s = ReplSession()
        result = s.eval_entry("param p: int = 7\nlet z: decimal = 1 / 0")
        assert not result.ok
        assert [(name, _int(value)) for name, _type, value in s.declared_params()] == [("p", 7)]

    def test_runtime_raise_excludes_param_declared_after_failure(self) -> None:
        # Regression: a runtime failure that precedes a later
        # ``param`` declaration must not record that param. The IR interpreter
        # installs every param into the base frame up front, so a naive
        # ``symbol in base frame`` check would record the later param even though
        # the scope-promotion loop excluded its binding by source position —
        # leaving ``declared_params()`` to raise ``KeyError``.
        s = ReplSession()
        result = s.eval_entry("let z: decimal = 1 / 0\nparam q: int = 5")
        assert not result.ok
        assert s.declared_params() == []
        # The later param was excluded from the session scope too.
        assert not s.eval_entry("q").ok

    def test_runtime_raise_does_not_install_failing_param_default(self) -> None:
        s = ReplSession()
        result = s.eval_entry("param p: decimal = 1 / 0")
        assert not result.ok
        assert s.declared_params() == []

    def test_failing_param_default_still_promotes_completed_type_with_no_function(self) -> None:
        # Regression: when the entry declares no zero-capture function closure,
        # a failing ``param`` default used to roll back EVERY declaration in the
        # entry, including a record type that fully completed before the param
        # was reached — an accidental gate on whether the entry happened to
        # declare a function, not on what actually finished.
        s = ReplSession()

        result = s.eval_entry("record Point\n  x: int\nparam p: int = [1, 2][9]")

        assert not result.ok
        followup = s.eval_entry("Point(x = 1)")
        assert followup.ok, followup.diagnostics

    def test_runtime_raise_does_not_promote_param_with_unpromoted_self_qualified_type(self) -> None:
        s = ReplSession()

        result = s.eval_entry(
            'param p: ::R\nlet stop: int = raise Abort(message = "stop")\nrecord R\n  value: int'
        )

        assert not result.ok
        assert s.declared_params() == []
        assert not s.eval_entry("p").ok

    def test_runtime_raise_promotes_preinstalled_function_before_failing_param_default(
        self,
    ) -> None:
        s = ReplSession()

        result = s.eval_entry("def f() -> int = 1\nparam p: decimal = 1 / 0")

        assert not result.ok
        assert result.installed == ("f",)
        assert s.eval_entry("f()").value == IntValue(1)

    def test_runtime_raise_does_not_promote_later_type(self) -> None:
        s = ReplSession()
        result = s.eval_entry("let z: decimal = 1 / 0\nrecord After\n  value: int")
        assert not result.ok
        assert not s.eval_entry("After").ok
        assert not s.eval_entry("After(value = 3)").ok

    def test_runtime_raise_does_not_promote_enum_without_its_later_referenced_member(
        self,
    ) -> None:
        s = ReplSession()

        failed = s.eval_entry("enum E = ::R\nlet z: decimal = 1 / 0\nrecord R()")

        assert not failed.ok
        assert "E" not in failed.installed
        assert "E" not in s.type_names()
        assert "R" not in s.type_names()

    def test_runtime_raise_tracks_applied_referenced_member_dependencies(self) -> None:
        s = ReplSession()

        failed = s.eval_entry(
            "enum E = ::R[Payload]\nlet z: decimal = 1 / 0\nrecord Payload()\nrecord R[T]()"
        )

        assert not failed.ok
        assert "E" not in s.type_names()

    def test_runtime_raise_does_not_promote_function_typed_with_later_inline_member(
        self,
    ) -> None:
        s = ReplSession()

        failed = s.eval_entry(
            "def read(value: E::A) -> int = value.value\n"
            "let z: decimal = 1 / 0\n"
            "enum E\n"
            "  | A(value: int)"
        )

        assert not failed.ok
        assert "read" not in failed.installed
        assert not s.eval_entry("read").ok

    def test_runtime_raise_retains_completed_function_initializer_metadata(self) -> None:
        s = ReplSession()
        failed = s.eval_entry("let z: decimal = 1 / 0\ndef later[T](x: T) -> T = x")

        assert not failed.ok
        assert s.eval_entry("later(1)").value_type == IntType()

    def test_runtime_failure_promotes_function_declared_after_failing_source_item(self) -> None:
        s = ReplSession()
        failed = s.eval_entry(
            'let before = 1\nlet stop: int = raise Abort(message = "stop")\ndef later() -> int = 4'
        )

        assert not failed.ok
        assert set(failed.installed) == {"before", "later"}
        assert s.eval_entry("later()").value == IntValue(4)
        assert not s.eval_entry("stop").ok

    def test_runtime_raise_does_not_promote_later_exception_type(self) -> None:
        s = ReplSession()
        result = s.eval_entry(
            "let z: decimal = 1 / 0\nexception Later extends Exception\n  code: int"
        )
        assert not result.ok
        assert not s.eval_entry("Later").ok
        assert not s.eval_entry('Later(message = "m", code = 3)').ok

    def test_runtime_raise_promotes_prior_exception_constructor(self) -> None:
        s = ReplSession()
        result = s.eval_entry(
            "exception Before extends Exception\n"
            "  code: int\n"
            "exception Child extends Before\n"
            "  detail: int\n"
            "let z: decimal = 1 / 0"
        )
        assert not result.ok
        assert s.eval_entry('Before(message = "m", code = 3)').ok
        assert s.eval_entry('Child(message = "m", code = 3, detail = 4)').ok

    def test_runtime_raise_preserves_prior_type_but_not_later_binding(self) -> None:
        s = ReplSession()
        result = s.eval_entry(
            "record Box\n  value: int\nlet z: decimal = 1 / 0\nlet after = 9\nafter"
        )
        assert not result.ok
        assert s.eval_entry("Box(value = 3)").ok
        assert not s.eval_entry("after").ok

    def test_runtime_raise_preserves_assign_to_prior_var(self) -> None:
        s = ReplSession()
        r1 = s.eval_entry("var v = 1")
        assert r1.ok
        r2 = s.eval_entry("v := 99\nlet z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["v"] == 99

    def test_runtime_raise_preserves_indexed_assign_to_prior_var(self) -> None:
        from agm.agl.semantics.values import ArrayValue, IntValue

        s = ReplSession()
        r1 = s.eval_entry("var xs = [1, 2, 3]")
        assert r1.ok
        r2 = s.eval_entry("xs[0] := 99\nlet z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        vals = {n: v for n, _t, v in s.bindings()}
        assert vals["xs"] == ArrayValue([IntValue(99), IntValue(2), IntValue(3)])

    def test_successful_assign_to_prior_var_persists(self) -> None:
        # The positive counterpart: a successful ``:=`` in a later entry DOES
        # persist into the session.
        s = ReplSession()
        s.eval_entry("var v = 1")
        r = s.eval_entry("v := 5")
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["v"] == 5

    def test_syntax_error_does_not_advance_state(self) -> None:
        s = ReplSession()
        r = s.eval_entry("let = = =")
        assert not r.ok
        assert r.diagnostics
        # A valid entry afterwards still works (node-id counter not advanced).
        r2 = s.eval_entry("let ok = 1")
        assert r2.ok


# ---------------------------------------------------------------------------
# Exactly-once agent dispatch
# ---------------------------------------------------------------------------


class TestExactlyOnce:
    def test_agent_fires_exactly_once(self) -> None:
        agent = CountingAgent("the-answer")
        s = ReplSession(agent_dispatcher=agent)
        r1 = s.eval_entry('let g = ask """say something"""')
        assert r1.ok
        assert agent.calls == 1
        # Referencing the stored binding in a LATER entry must NOT re-invoke.
        r2 = s.eval_entry("g")
        assert r2.ok
        assert _text(r2.value) == "the-answer"
        assert agent.calls == 1

    def test_standalone_ask_echo_is_unquoted(self) -> None:
        from agm.agl.repl.render import render_entry_result

        agent = CountingAgent("the-answer")
        s = ReplSession(agent_dispatcher=agent)
        result = s.eval_entry('ask """say something"""')

        assert result.ok
        assert result.quote_strings is False
        assert render_entry_result(result, echo=True) == "the-answer"

    def test_stored_ask_result_echo_uses_normal_text_quoting(self) -> None:
        from agm.agl.repl.render import render_entry_result

        agent = CountingAgent("the-answer")
        s = ReplSession(agent_dispatcher=agent)
        first = s.eval_entry('let txt: text = ask """say something"""')
        second = s.eval_entry("txt")

        assert first.ok
        assert first.quote_strings is True
        assert second.ok
        assert second.quote_strings is True
        assert render_entry_result(second, echo=True) == '"the-answer"'

    def test_distinct_agent_responses_across_entries(self) -> None:
        agent = CountingAgent("first", "second", "third")
        s = ReplSession(agent_dispatcher=agent)
        s.eval_entry('let a = ask """q1"""')
        s.eval_entry('let b = ask """q2"""')
        s.eval_entry('let c = ask """q3"""')
        vals = {n: _text(v) for n, _t, v in s.bindings()}
        assert vals == {"a": "first", "b": "second", "c": "third"}
        assert agent.calls == 3

    def test_agent_value_dispatch(self) -> None:
        named = CountingAgent("named-reply")
        s = ReplSession(agent_dispatcher=named)
        r = s.eval_entry(
            'let reviewer = AgentCommand("reviewer")\n'
            'let out = ask("""review this""", agent = reviewer)'
        )
        assert r.ok, r.diagnostics
        assert _text({name: value for name, _typ, value in s.bindings()}["out"]) == "named-reply"
        assert named.calls == 1


# ---------------------------------------------------------------------------
# Agent declarations / ambient registration (host registration declares+backs)
# ---------------------------------------------------------------------------


class TestAgentDeclarations:
    def test_agent_value_dispatches_without_a_declaration(self) -> None:
        s = ReplSession(agent_dispatcher=CountingAgent("ok"))
        r = s.eval_entry(
            'let reviewer = AgentCommand("reviewer")\nask("""look""", agent = reviewer)'
        )
        assert r.ok

    def test_undeclared_unregistered_agent_call_errors(self) -> None:
        # A call to an agent that is neither registered nor declared in source is
        # still a static scope binding error.
        s = ReplSession()
        r = s.eval_entry('ghost "hi"')
        assert not r.ok
        assert r.diagnostics

    def test_cross_entry_agent_value_resolves(self) -> None:
        s = ReplSession(agent_dispatcher=CountingAgent("done"))
        r1 = s.eval_entry('let helper = AgentCommand("helper")')
        assert r1.ok
        r2 = s.eval_entry('let out = ask("""go""", agent = helper)')
        assert r2.ok, r2.diagnostics
        assert _text({name: value for name, _typ, value in s.bindings()}["out"]) == "done"

    def test_scoped_declaration_retains_its_handle_across_entries(self) -> None:
        agent = CountingAgent("done")
        s = ReplSession(agent_dispatcher=agent)

        declared = s.eval_entry('scope Tools\nlet helper = AgentCommand("helper")\nend Tools')
        assert declared.ok, declared.diagnostics
        ref = s._session_scope_nodes[("Tools",)].members["helper"]
        handle = s._link_image.symbol_for_decl(ref.decl_node_id)
        assert handle is not None

        called = s.eval_entry('let out = ask("""go""", agent = Tools::helper)')

        assert called.ok, called.diagnostics
        assert s._link_image.symbol_for_decl(ref.decl_node_id) == handle
        assert _text({name: value for name, _typ, value in s.bindings()}["out"]) == "done"
        assert agent.calls == 1

    def test_failed_entry_declaration_does_not_persist(self) -> None:
        # A declaration in an entry that fails to promote must NOT leak into the
        # ambient set: a later call relying on it is still a scope error.
        s = ReplSession()
        # The entry declares ``maybe`` but then has a type error, so it fails and
        # rolls back; the declaration must not persist.
        bad = s.eval_entry('agent maybe\nlet x: int = "oops"')
        assert not bad.ok
        r = s.eval_entry('maybe "call"')
        assert not r.ok
        assert r.diagnostics

    def test_type_of_allows_agent_value_call(self) -> None:
        s = ReplSession()
        s.eval_entry('let reviewer = AgentCommand("reviewer")')
        assert s.type_of('ask("""ask""", agent = reviewer)') == repr(TextType())

    def test_reset_clears_declared_agents(self) -> None:
        # After reset, a previously source-declared agent is gone: a call to it
        # (without re-registration/re-declaration) is a scope error again.
        s = ReplSession()
        s.eval_entry("agent transient")
        s.reset()
        r = s.eval_entry('transient "hi"')
        assert not r.ok


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TestParams:
    def test_declared_param_listed_unset(self) -> None:
        s = ReplSession()
        s.eval_entry('param name: text = "World"')
        ins = s.declared_params()
        assert len(ins) == 1
        name, typ, val = ins[0]
        assert name == "name"
        assert isinstance(typ, TextType)
        assert _text(val) == "World"

    def test_unset_param_reference_is_clean_error(self) -> None:
        s = ReplSession()
        r = s.eval_entry("param name: text")
        assert not r.ok
        assert r.diagnostics
        assert "name" in r.diagnostics[0].message
        assert "Missing required param" in r.diagnostics[0].message

    def test_unset_param_diagnostic_names_the_prompt_entry_namespace(self) -> None:
        """The prompt has no module name, so its params read as ``@entry::``.

        The entry module's internal sentinel is not a spelling a user can type
        and must never reach a diagnostic.
        """
        s = ReplSession()

        r = s.eval_entry("param name: text")

        assert not r.ok
        assert "'@entry::name'" in r.diagnostics[0].message
        assert "<entry>" not in r.diagnostics[0].message

    def test_declared_param_then_reference(self) -> None:
        s = ReplSession()
        s.eval_entry('param name: text = "World"')
        r = s.eval_entry("name")
        assert r.ok
        assert _text(r.value) == "World"
        _n, _t, val = s.declared_params()[0]
        assert val is not None

    def test_imported_required_param_fails_before_interpreter_creation(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "settings.agl").write_text("param token: text\n")
        session = ReplSession(lib_root=tmp_path, default_stdlib=False)

        result = session.eval_entry("import settings\n()")

        assert not result.ok
        assert "Missing required param" in result.diagnostics[0].message
        assert "token" in result.diagnostics[0].message

    def test_import_cycle_param_default_dependency_cycle_is_diagnostic(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.agl").write_text(
            "import b\nparam a_value: int = b::read()\ndef read() -> int = a_value\n"
        )
        (tmp_path / "b.agl").write_text(
            "import a\nparam b_value: int = a::read()\ndef read() -> int = b_value\n"
        )
        session = ReplSession(lib_root=tmp_path, default_stdlib=False)

        result = session.eval_entry("import a\n()")

        assert not result.ok
        assert result.error is None
        assert len(result.diagnostics) == 1
        assert "parameter defaults form a dependency cycle" in result.diagnostics[0].message.lower()

    def test_later_import_validates_active_params_without_reinstalling_them(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "settings.agl").write_text(
            "param items: array[int]\n"
            "def update() -> unit =\n"
            "  items[0] := 9\n"
            "def read() -> int = items[0]\n"
        )
        (tmp_path / "other.agl").write_text("param count: int = 0\n")
        seen_inventories: list[tuple[str, ...]] = []

        def load_config(params: tuple[IrParam, ...]) -> dict[str, object]:
            seen_inventories.append(tuple(param.qualified_public_name for param in params))
            return {"settings::items": [1, 2]}

        session = ReplSession(
            lib_root=tmp_path,
            default_stdlib=False,
            params_config_loader=load_config,
        )

        assert session.eval_entry("import settings\nsettings::update()").ok
        later = session.eval_entry("import other\nsettings::read()")

        assert later.ok, later.diagnostics
        assert later.value == IntValue(9)
        assert seen_inventories == [
            ("settings::items",),
            ("settings::items", "other::count"),
        ]

    def test_partial_promotion_retains_completed_imported_param_inventory(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "settings.agl").write_text(
            "param token: array[text]\n"
            "def update() -> unit =\n"
            '  token[0] := "mutated"\n'
            "def read() -> text = token[0]\n"
        )
        (tmp_path / "other.agl").write_text("param count: int = 0\n")
        seen_inventories: list[tuple[str, ...]] = []

        def load_config(params: tuple[IrParam, ...]) -> dict[str, object]:
            seen_inventories.append(tuple(param.qualified_public_name for param in params))
            return {"settings::token": ["configured"]}

        session = ReplSession(
            lib_root=tmp_path,
            default_stdlib=False,
            params_config_loader=load_config,
        )

        failed = session.eval_entry(
            "import settings\n"
            "def retained() -> text = settings::read()\n"
            "settings::update()\n"
            '"bad" as int'
        )
        assert not failed.ok
        assert failed.installed == ("retained",), (failed.diagnostics, failed.error)

        later = session.eval_entry("import other\nretained()")
        reimported = session.eval_entry("import settings\nsettings::read()")

        assert later.ok, later.diagnostics
        assert later.value == TextValue("mutated")
        assert reimported.ok, reimported.diagnostics
        assert reimported.value == TextValue("mutated")
        assert seen_inventories == [
            ("settings::token",),
            ("settings::token", "other::count"),
            ("settings::token", "other::count"),
        ]

    def test_partial_failure_does_not_retain_params_or_dependents_of_incomplete_module(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "broken.agl").write_text(
            'param ready: text = "ready"\nparam failed: int = "bad" as int\n'
        )
        (tmp_path / "unfinished.agl").write_text("let value: int = 1\n")
        (tmp_path / "wrapper.agl").write_text(
            "import broken\nimport unfinished\ndef noop() -> unit = ()\n"
        )
        seen_inventories: list[tuple[str, ...]] = []

        def load_config(params: tuple[IrParam, ...]) -> dict[str, object]:
            seen_inventories.append(tuple(param.qualified_public_name for param in params))
            return {}

        session = ReplSession(
            lib_root=tmp_path,
            default_stdlib=False,
            params_config_loader=load_config,
        )

        failed = session.eval_entry("import wrapper\n()")

        assert not failed.ok
        assert seen_inventories == [("broken::ready", "broken::failed")]
        assert session._active_imported_params == {}
        assert session._loaded_lib_modules == {}
        assert session._link_image._linked_modules == set()

    def test_declared_param_typed_value(self) -> None:
        s = ReplSession()
        s.eval_entry("param count: int = 42")
        r = s.eval_entry("count + 1")
        assert r.ok
        assert _int(r.value) == 43

    def test_param_default_is_in_bindings(self) -> None:
        s = ReplSession()
        s.eval_entry('param name: text = "hi"')
        assert any(n == "name" for n, _t, _v in s.bindings())

    def test_unset_scoped_param_reference_is_clean_error(self) -> None:
        # A required scoped param must report the same clean diagnostic as a
        # root param and leave the session alive for later entries, not crash
        # with an unhandled IR error.
        s = ReplSession()
        r = s.eval_entry("scope A\nparam p: int\nend A\nprint(A::p)")
        assert not r.ok
        assert r.diagnostics
        assert "A::p" in r.diagnostics[0].message
        assert "Missing required param" in r.diagnostics[0].message
        after = s.eval_entry('"still alive"')
        assert after.ok
        assert _text(after.value) == "still alive"

    def test_declared_params_lists_scoped_param_by_full_path(self) -> None:
        s = ReplSession()
        s.eval_entry("scope A\nparam p: int = 5\nend A")
        ins = s.declared_params()
        assert len(ins) == 1
        name, typ, val = ins[0]
        assert name == "A::p"
        assert isinstance(typ, IntType)
        assert _int(val) == 5

    def test_scoped_param_metadata_is_removed_when_another_member_replaces_it(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope A\nparam x: int = 1\nend A").ok

        replacement = s.eval_entry("scope A\nlet x = 2\nend A")

        assert replacement.ok, replacement.diagnostics
        assert s.declared_params() == []
        value = s.eval_entry("A::x")
        assert value.ok, value.diagnostics
        assert value.value == IntValue(2)

    def test_let_binding_displaces_same_named_root_param_in_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("param count: int = 1").ok

        result = s.eval_entry("let count = 5")

        assert result.ok, result.diagnostics
        assert s.declared_params() == []
        value = s.eval_entry("count")
        assert value.ok, value.diagnostics
        assert value.value == IntValue(5)

    def test_let_binding_displaces_same_keyed_scoped_param_in_later_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry('scope Deploy\nparam region: text = "eu"\nend Deploy').ok

        result = s.eval_entry('scope Deploy\nlet region = "us"\nend Deploy')

        assert result.ok, result.diagnostics
        assert s.declared_params() == []
        value = s.eval_entry("Deploy::region")
        assert value.ok, value.diagnostics
        assert value.value == TextValue("us")

    def test_single_entry_displaces_root_and_scoped_param_together(self) -> None:
        s = ReplSession()
        setup = s.eval_entry(
            'param count: int = 1\nscope Deploy\nparam region: text = "eu"\nend Deploy'
        )
        assert setup.ok, setup.diagnostics
        assert {name for name, _t, _v in s.declared_params()} == {"count", "Deploy::region"}

        result = s.eval_entry('let count = 5\nscope Deploy\nlet region = "us"\nend Deploy')

        assert result.ok, result.diagnostics
        assert s.declared_params() == []

    def test_undisplaced_param_remains_listed_after_sibling_displacement(self) -> None:
        s = ReplSession()
        setup = s.eval_entry("param count: int = 1\nparam other: int = 2")
        assert setup.ok, setup.diagnostics

        result = s.eval_entry("let count = 5")

        assert result.ok, result.diagnostics
        assert {name for name, _t, _v in s.declared_params()} == {"other"}

    def test_scoped_param_failing_default_does_not_corrupt_next_entry(self) -> None:
        # A scoped param whose default raises must not be promoted; the next
        # entry must degrade gracefully rather than crash on an unbound symbol.
        s = ReplSession()
        first = s.eval_entry('scope A\nparam p: int = "x" as int\nend A')
        assert not first.ok
        assert s.declared_params() == []
        second = s.eval_entry("let q = A::p + 1")
        assert not second.ok
        assert second.diagnostics


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        s.eval_entry("param n: int")
        s.reset()
        assert s.bindings() == []
        assert s.declared_params() == []
        assert s.dump_source() == ""
        # After reset a name previously defined is gone (would error on ref).
        r = s.eval_entry("x")
        assert not r.ok

    def test_reset_restarts_node_ids(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 1")
        s.reset()
        r = s.eval_entry("let a = 2")
        assert r.ok
        assert _int({name: value for name, _typ, value in s.bindings()}["a"]) == 2

    def test_reset_clears_retained_uses(self) -> None:
        s = ReplSession()
        assert s.eval_entry("scope Tools\ndef twice(x: int) -> int = x * 2\nend Tools").ok
        assert s.eval_entry("use Tools::*").ok
        assert s.eval_entry("twice(3)").ok

        s.reset()

        assert s.eval_entry("scope Tools\ndef twice(x: int) -> int = x * 2\nend Tools").ok
        result = s.eval_entry("twice(3)")
        assert not result.ok


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
    @pytest.mark.parametrize(
        "transcript",
        (
            "# agm:repl-transcript:v1\ninvalid",
            "# agm:repl-transcript:v1\n# agm:entry:1",
            "# agm:repl-transcript:v1\n# agm:entry:x\n",
            "# agm:repl-transcript:v1\n# agm:entry:2\nx\n",
        ),
    )
    def test_malformed_saved_transcript_is_treated_as_an_ordinary_file(
        self, transcript: str
    ) -> None:
        from agm.agl.repl.session import _decode_transcript

        assert _decode_transcript(transcript) is None

    def test_load_file_executes_into_session(self, tmp_path: Path) -> None:
        f = tmp_path / "prog.agl"
        f.write_text("let a = 1\nlet b = a + 2\n")
        s = ReplSession()
        results = s.load_file(f)
        assert all(r.ok for r in results)
        assert len(results) == 2
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals == {"a": 1, "b": 3}

    def test_load_file_agent_runs_once(self, tmp_path: Path) -> None:
        agent = CountingAgent("loaded")
        f = tmp_path / "p.agl"
        f.write_text('let g = ask """hi"""\n')
        s = ReplSession(agent_dispatcher=agent)
        s.load_file(f)
        assert agent.calls == 1
        # Referencing it later does not re-run.
        s.eval_entry("g")
        assert agent.calls == 1

    def test_saved_transcript_halts_at_first_failed_entry(self, tmp_path: Path) -> None:
        transcript = tmp_path / "failed.agl"
        transcript.write_text(
            "# agm:repl-transcript:v1\n"
            "# agm:entry:9\nlet a = 1\n"
            "# agm:entry:7\nmissing\n"
            "# agm:entry:9\nlet b = 2\n",
            encoding="utf-8",
        )

        session = ReplSession()
        results = session.load_file(transcript)

        assert len(results) == 2
        assert results[0].ok
        assert not results[1].ok
        assert [name for name, _type, _value in session.bindings()] == ["a"]

    def test_load_file_incremental_redefinition_round_trips(self, tmp_path: Path) -> None:
        # Redefinition across entries is supported; a saved transcript containing
        # a redefinition must reload because :load runs one statement per entry.
        a = ReplSession()
        a.eval_entry("let x = 1")
        a.eval_entry("let x = 2")
        f = tmp_path / "redef.agl"
        f.write_text(a.dump_source())

        b = ReplSession()
        results = b.load_file(f)
        assert all(r.ok for r in results)
        vals = {n: _int(v) for n, _t, v in b.bindings()}
        assert vals == {"x": 2}

    def test_load_file_round_trips_a_forward_resolved_use_entry(self, tmp_path: Path) -> None:
        original = ReplSession()
        assert original.eval_entry("use S::*\ndef S::value() -> int = 1").ok
        transcript = tmp_path / "session.agl"
        transcript.write_text(original.dump_source(), encoding="utf-8")

        loaded = ReplSession()
        results = loaded.load_file(transcript)

        assert all(result.ok for result in results)
        value = loaded.eval_entry("value()")
        assert value.ok, value.diagnostics
        assert value.value == IntValue(1)

    def test_load_file_round_trips_a_use_after_a_declaration(self, tmp_path: Path) -> None:
        original = ReplSession()
        assert original.eval_entry("scope S\ndef value() -> int = 1\nend S").ok
        assert original.eval_entry("use S::*").ok
        transcript = tmp_path / "session.agl"
        transcript.write_text(original.dump_source(), encoding="utf-8")

        loaded = ReplSession()
        results = loaded.load_file(transcript)

        assert all(result.ok for result in results)
        value = loaded.eval_entry("value()")
        assert value.ok, value.diagnostics
        assert value.value == IntValue(1)

    def test_load_file_can_be_loaded_twice_with_nominal_and_function_definitions(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "functions.agl"
        f.write_text(
            "enum Counter = zero\n"
            "def increment(n: int) = n + 1\n"
            "def reset(_: Counter) = Counter::zero\n"
        )
        session = ReplSession()

        first = session.load_file(f)
        second = session.load_file(f)

        assert all(result.ok for result in (*first, *second))
        result = session.eval_entry("increment(2)")
        assert result.ok
        assert _int(result.value) == 3

    def test_load_file_multi_binding_round_trips(self, tmp_path: Path) -> None:
        a = ReplSession()
        a.eval_entry("let a = 1")
        a.eval_entry("let b = a + 1")
        f = tmp_path / "multi.agl"
        f.write_text(a.dump_source())

        b = ReplSession()
        results = b.load_file(f)
        assert all(r.ok for r in results)
        vals = {n: _int(v) for n, _t, v in b.bindings()}
        assert vals == {"a": 1, "b": 2}

    def test_load_file_block_statement_slices_correctly(self, tmp_path: Path) -> None:
        # A multi-line block statement must be sliced with its nested indentation
        # preserved so each top-level slice is independently parseable.
        f = tmp_path / "block.agl"
        f.write_text(
            "let n = 1\n"
            'var label: text = ""\n'
            "if n == 1 =>\n"
            '  label := "one"\n'
            "| else =>\n"
            '  label := "many"\n'
            "label\n"
        )
        s = ReplSession()
        results = s.load_file(f)
        assert all(r.ok for r in results), [r.diagnostics for r in results if not r.ok]
        vals = {n: v for n, _t, v in s.bindings()}
        assert _text(vals["label"]) == "one"

    def test_load_file_record_block_slices_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "rec.agl"
        f.write_text("record Point\n    x: int\n    y: int\nlet p = Point(x = 1, y = 2)\np.x\n")
        s = ReplSession()
        results = s.load_file(f)
        assert all(r.ok for r in results), [r.diagnostics for r in results if not r.ok]
        assert results[-1].value is not None
        assert _int(results[-1].value) == 1

    def test_load_file_halts_at_first_error_keeps_prior(self, tmp_path: Path) -> None:
        f = tmp_path / "halt.agl"
        f.write_text(
            "let a = 1\n"
            "let z: decimal = 1 / 0\n"  # runtime raise — halts the load here
            "let b = 99\n"  # never reached
        )
        s = ReplSession()
        results = s.load_file(f)
        # The load halted at the failing statement; nothing after it ran.
        assert len(results) == 2
        assert results[0].ok
        assert not results[1].ok
        # The statement before the failure persisted; 'b' never ran.
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals == {"a": 1}

    def test_load_file_syntax_error_single_failed_result(self, tmp_path: Path) -> None:
        f = tmp_path / "syntax.agl"
        f.write_text("let = oops\n")
        s = ReplSession()
        results = s.load_file(f)
        assert len(results) == 1
        assert not results[0].ok
        assert results[0].diagnostics

    def test_load_file_empty_file_no_results(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.agl"
        f.write_text("")
        s = ReplSession()
        results = s.load_file(f)
        assert results == []
        assert s.bindings() == []

    def test_load_file_comment_only_no_results(self, tmp_path: Path) -> None:
        f = tmp_path / "comments.agl"
        f.write_text("# just a comment\n# and another\n")
        s = ReplSession()
        results = s.load_file(f)
        assert results == []
        assert s.bindings() == []


# ---------------------------------------------------------------------------
# dump_source
# ---------------------------------------------------------------------------


class TestDumpSource:
    def test_dump_source_accumulates_successful_entries(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 1")
        s.eval_entry("let b = 2")
        assert s.dump_source() == (
            "# agm:repl-transcript:v1\n# agm:entry:9\nlet a = 1\n# agm:entry:9\nlet b = 2\n"
        )

    def test_dump_source_excludes_failed_entries(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 1")
        s.eval_entry("let z: decimal = 1 / 0")  # runtime fail
        s.eval_entry('let b = a + "x"')  # type fail
        assert s.dump_source() == ("# agm:repl-transcript:v1\n# agm:entry:9\nlet a = 1\n")


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_non_exhaustive_case_error_surfaced(self) -> None:
        s = ReplSession()
        s.eval_entry("enum R\n  | Pass\n  | Fail")
        s.eval_entry("let r: R = Pass")
        r = s.eval_entry("case r of\n  | Pass() => ()")
        assert not r.ok
        assert len(r.diagnostics) == 1
        assert "Fail" in r.diagnostics[0].message
        assert r.warnings == []

    def test_tab_warning_surfaced(self) -> None:
        # A TAB character in the entry source surfaces a per-entry advisory
        # warning (mirroring ``PipelineDriver.run``), without failing the entry.
        s = ReplSession()
        r = s.eval_entry("let x =\t1")
        assert r.ok
        assert any("TAB" in w.message or "tab" in w.message for w in r.warnings)

    def test_match_error_on_check_only_path(self) -> None:
        s = ReplSession()
        s.eval_entry("enum R\n  | Pass\n  | Fail")
        s.eval_entry("let r: R = Pass")
        r = s.eval_entry("case r of\n  | Pass() => ()", check_only=True)
        assert not r.ok
        assert len(r.diagnostics) == 1
        assert r.warnings == []


# ---------------------------------------------------------------------------
# check_only
# ---------------------------------------------------------------------------


class TestCheckOnly:
    def test_check_only_types_expression_without_eval(self) -> None:
        agent = CountingAgent("nope")
        s = ReplSession(agent_dispatcher=agent)
        r = s.eval_entry('ask """ask"""', check_only=True)
        assert r.ok
        assert r.kind == "expression"
        assert isinstance(r.value_type, TextType)
        assert r.value is None
        assert agent.calls == 0

    def test_check_only_does_not_promote(self) -> None:
        s = ReplSession()
        r = s.eval_entry("let x = 1", check_only=True)
        assert r.ok
        assert r.kind == "binding"
        assert r.name == "x"
        assert isinstance(r.value_type, IntType)
        assert r.value is None
        # Not promoted: a later reference fails.
        assert s.bindings() == []
        assert not s.eval_entry("x").ok

    def test_check_only_trailing_binder_reports_bottom_initializer(self) -> None:
        s = ReplSession()
        r = s.eval_entry('let x: int = raise Abort(message = "x")', check_only=True)

        assert r.ok
        assert r.kind == "binding"
        assert r.name == "x"
        assert isinstance(r.value_type, BottomType)
        assert r.value is None

    def test_check_only_does_not_advance_node_ids(self) -> None:
        s = ReplSession()
        s.eval_entry("check_only", check_only=True)  # statement-ish; ignored result
        # A real binding after a check_only still works.
        r = s.eval_entry("let a = 1")
        assert r.ok

    def test_check_only_declaration_kind(self) -> None:
        s = ReplSession()
        r = s.eval_entry("record P\n  x: int", check_only=True)
        assert r.ok
        assert r.kind == "declaration"
        assert r.name == "P"
        # Not promoted.
        assert not s.eval_entry("let p = P(x = 1)").ok

    def test_check_only_type_error_still_fails(self) -> None:
        s = ReplSession()
        s.eval_entry('let t = "x"')
        r = s.eval_entry("t + 1", check_only=True)
        assert not r.ok
        assert r.diagnostics


# ---------------------------------------------------------------------------
# Registration / agents listing
# ---------------------------------------------------------------------------


class TestRegistrationAndAgents:
    def test_register_codec_validation(self) -> None:
        from agm.agl.runtime.codec import JsonCodec

        s = ReplSession()
        with pytest.raises(ValueError):
            s.register_codec(JsonCodec())  # reserved built-in name


# ---------------------------------------------------------------------------
# EntryResult shape
# ---------------------------------------------------------------------------


class TestContractError:
    def test_contract_materialization_error_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object, type_table: object = None) -> object:
            raise ValueError("bad contract")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)
        from agm.agl.runtime.codec import TextCodec

        class BadCodec(TextCodec):
            @property
            def name(self) -> str:
                return "bad"

        s = ReplSession(agent_dispatcher=CountingAgent("ok"))
        s.register_codec(BadCodec())
        r = s.eval_entry('let x = ask("hi", format = "bad")')
        assert not r.ok
        assert any("Contract error" in d.message for d in r.diagnostics)
        # Atomic: nothing promoted.
        assert s.bindings() == []


class TestEntryResultShape:
    def test_result_is_frozen_dataclass(self) -> None:
        s = ReplSession()
        r = s.eval_entry("let x = 1")
        assert isinstance(r, EntryResult)
        assert r.trace_path is None  # no --log-file → no trace path
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.ok = False


# ---------------------------------------------------------------------------
# Agent-call cancellation (declined / interrupted)
# ---------------------------------------------------------------------------


class _CancellingAgent:
    """A fake ``AgentFn`` that raises ``AgentCancelled`` on dispatch."""

    def __init__(self, callee: str = "ask", reason: str = "declined") -> None:
        self._callee = callee
        self._reason = reason
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        from agm.agl.repl.agents import AgentCancelled

        self.calls += 1
        raise AgentCancelled(self._callee, self._reason)


class _InterruptAgent:
    """A fake ``AgentFn`` that raises a bare ``KeyboardInterrupt`` (Ctrl-C)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
        raise KeyboardInterrupt


class TestAgentCancellation:
    def test_interrupt_without_agent_reports_interrupted_entry(self) -> None:
        session = ReplSession()
        with patch("agm.agl.eval.ir_interpreter.IrInterpreter.run", side_effect=KeyboardInterrupt):
            result = session.eval_entry("1 + 1")

        assert not result.ok
        assert result.error is None
        assert result.diagnostics
        message = result.diagnostics[0].message.lower()
        assert "interrupted" in message
        assert "agent call" not in message
        assert session.eval_entry("2 + 2").ok

    def test_declined_agent_aborts_entry_with_diagnostic(self) -> None:
        s = ReplSession(agent_dispatcher=_CancellingAgent())
        r = s.eval_entry('let g = ask """do it"""')
        assert not r.ok
        assert r.error is None
        assert r.diagnostics
        assert "cancelled" in r.diagnostics[0].message.lower()

    def test_declined_agent_leaves_bindings_unchanged(self) -> None:
        s = ReplSession(agent_dispatcher=_CancellingAgent())
        s.eval_entry("let keep = 7")
        before = _snapshot(s)
        r = s.eval_entry('let g = ask """do it"""')
        assert not r.ok
        # The cancelled initializer did not complete, so it installs nothing.
        assert _snapshot(s) == before
        assert all(n != "g" for n, _t, _v in s.bindings())

    def test_keyboard_interrupt_aborts_entry(self) -> None:
        s = ReplSession(agent_dispatcher=_InterruptAgent())
        s.eval_entry("let x = 1")
        before = _snapshot(s)
        r = s.eval_entry('let g = ask """slow"""')
        assert not r.ok
        assert r.error is None
        assert _snapshot(s) == before

    def test_cancellation_preserves_prior_assignment(self) -> None:
        s = ReplSession(agent_dispatcher=_CancellingAgent())
        s.eval_entry("var v = 1")
        r = s.eval_entry('v := 2\nlet g = ask """x"""')
        assert not r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["v"] == 2

    def test_cancellation_preserves_completed_record(self) -> None:
        # Regression: a record (or enum / type alias) declared
        # before a cancelled agent call must be promoted, mirroring the
        # partial-effects behavior for runtime raises. Previously cancellation
        # carried no failure span, so every type declaration was dropped.
        s = ReplSession(agent_dispatcher=_CancellingAgent())
        r = s.eval_entry('record Box\n  value: int\nlet g = ask """x"""')
        assert not r.ok
        assert s.eval_entry("Box(value = 3)").ok

    def test_cancellation_excludes_record_declared_after_call(self) -> None:
        # A type declared after the cancelled call is not promoted.
        s = ReplSession(agent_dispatcher=_CancellingAgent())
        r = s.eval_entry('let g = ask """x"""\nrecord After\n  value: int')
        assert not r.ok
        assert not s.eval_entry("After(value: 1)").ok


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------


class TestTraceLogging:
    def test_no_trace_path_writes_nothing(self, tmp_path: Path) -> None:
        s = ReplSession(agent_dispatcher=CountingAgent("ok"))
        r = s.eval_entry('let g = ask """hi"""')
        assert r.ok
        assert r.trace_path is None

    def test_trace_file_records_run_and_agent_call(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(agent_dispatcher=CountingAgent("reply"), trace_path=trace)
        r = s.eval_entry('let g = ask """ask"""')
        assert r.ok
        assert r.trace_path == trace
        assert trace.exists()
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        kinds = [rec["kind"] for rec in records]
        assert "run_start" in kinds
        assert "run_end" in kinds
        assert "agent_request" in kinds
        assert "agent_response" in kinds

    def test_each_entry_is_its_own_run(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(agent_dispatcher=CountingAgent("a", "b"), trace_path=trace)
        s.eval_entry('let x = ask """one"""')
        s.eval_entry('let y = ask """two"""')
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        run_ids = {rec["run_id"] for rec in records}
        # Per-entry TraceStore → a fresh run_id per entry, all in one file.
        assert len(run_ids) == 2

    def test_check_only_writes_no_trace(self, tmp_path: Path) -> None:
        trace = tmp_path / "repl.log"
        s = ReplSession(agent_dispatcher=CountingAgent("ok"), trace_path=trace)
        r = s.eval_entry('let g = ask """hi"""', check_only=True)
        assert r.ok
        assert r.trace_path is None
        assert not trace.exists()

    def test_cancelled_entry_records_run_end(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(agent_dispatcher=_CancellingAgent(), trace_path=trace)
        r = s.eval_entry('let g = ask """x"""')
        assert not r.ok
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        run_end = [rec for rec in records if rec["kind"] == "run_end"]
        assert run_end and run_end[-1]["ok"] is False
        assert any(rec["kind"] == "agent_request" for rec in records)
        responses = [rec for rec in records if rec["kind"] == "agent_response"]
        assert responses and responses[-1]["cancelled"] is True
        assert responses[-1]["reason"]

    def test_write_failure_disables_logging_for_that_entry_only(self, tmp_path: Path) -> None:
        """A transient write failure must not kill logging for the whole session."""
        import json

        from agm.core import log as core_log

        trace = tmp_path / "repl.log"
        s = ReplSession(agent_dispatcher=CountingAgent("a", "b"), trace_path=trace)
        real_append = core_log.append_jsonl
        calls = {"n": 0}

        def flaky(path: Path | None, record: Mapping[str, object]) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("transient failure")
            real_append(path, record)

        with patch("agm.agl.runtime.trace.append_jsonl", side_effect=flaky):
            first = s.eval_entry('print "one"')
            second = s.eval_entry('print "two"')

        assert first.ok
        assert second.ok
        assert second.trace_path == trace
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        rendered = [rec["rendered"] for rec in records if rec["kind"] == "print"]
        assert rendered == ["two"]


# ---------------------------------------------------------------------------
# Removed legacy preset API
# ---------------------------------------------------------------------------


class TestRemovedPresetParam:
    def test_reset_keeps_declared_params_empty(self) -> None:
        s = ReplSession()
        s.eval_entry("param count: int = 42")
        s.reset()
        assert s.declared_params() == []


# ---------------------------------------------------------------------------
# Issue #1 — re-declared param: stale value must be purged from value scope
# ---------------------------------------------------------------------------


class TestParamRedeclaration:
    def test_redeclare_param_purges_stale_value_from_bindings(self) -> None:
        s = ReplSession()
        r1 = s.eval_entry("param x: int = 5")
        assert r1.ok
        r2 = s.eval_entry("param x: int = 10")
        assert r2.ok
        ins2 = {name: val for name, _t, val in s.declared_params()}
        assert _int(ins2["x"]) == 10

    def test_redeclare_param_then_reference_raises_unset_guard(self) -> None:
        s = ReplSession()
        s.eval_entry("param x: int = 5")
        s.eval_entry("param x: int = 10")
        r = s.eval_entry("x + 1")
        assert r.ok
        assert _int(r.value) == 11

    def test_redeclare_param_then_reset_works(self) -> None:
        s = ReplSession()
        s.eval_entry("param x: int = 5")
        s.eval_entry("param x: int = 10")
        r = s.eval_entry("x + 1")
        assert r.ok
        assert _int(r.value) == 11


# ---------------------------------------------------------------------------
# Issue #7 — snapshot optimisation: assignment to a prior binding still rolls back
# ---------------------------------------------------------------------------


class TestSnapshotOptimisation:
    def test_assign_to_prior_binding_in_raising_entry_persists(self) -> None:
        s = ReplSession()
        r1 = s.eval_entry("var counter = 0")
        assert r1.ok
        # This entry assigns counter=99 then raises (division by zero).
        r2 = s.eval_entry("counter := 99\nlet _z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        # Completed effects remain visible after a later initializer raises.
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["counter"] == 99

    def test_entry_without_assignment_does_not_corrupt_prior_bindings(self) -> None:
        """An entry with no ``:=`` statements leaves prior bindings untouched.

        This guards that the optimisation (no snapshot for assignment-free entries) does
        not accidentally allow prior bindings to be mutated on success.
        """
        s = ReplSession()
        s.eval_entry("var a = 1")
        s.eval_entry("let b = 2")
        # An entry that only reads a and b, with no assignment.
        r = s.eval_entry("a + b")
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["a"] == 1
        assert vals["b"] == 2

    def test_entry_with_only_new_bindings_does_not_disturb_prior(self) -> None:
        """Adding new bindings in an entry that raises leaves old bindings clean."""
        s = ReplSession()
        s.eval_entry("let x = 10")
        # Entry raises; it tries to add a new binding (no assignment to prior state).
        r = s.eval_entry("let _fail: decimal = 1 / 0")
        assert not r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals == {"x": 10}  # x untouched; _fail never promoted


# ---------------------------------------------------------------------------
# if-expression in the REPL
# ---------------------------------------------------------------------------


class TestIfExpr:
    def test_parenthesized_if_expr_echoes_value(self) -> None:
        # A parenthesized if-expression at the prompt wraps into an ExprStmt,
        # so _classify returns "expression" and the evaluated value is echoed.
        s = ReplSession()
        r = s.eval_entry("(if true => 1 | else => 2)")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 1

    def test_parenthesized_if_expr_else_branch_taken(self) -> None:
        # Verify the else branch is taken when the condition is false.
        s = ReplSession()
        r = s.eval_entry("(if false => 1 | else => 2)")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 2

    def test_parenthesized_if_expr_leading_pipe_echoes_value(self) -> None:
        # The leading-pipe form inside parens also works as an expression echo.
        s = ReplSession()
        r = s.eval_entry("(if | true => 10 | else => 20)")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 10

    def test_bare_if_expr_classified_as_expression(self) -> None:
        # In AgL, ``if`` is a value-producing expression.  A bare ``if`` entry
        # at the prompt is classified as "expression" (it yields a value).
        # The value is void when the branches are statement-like (e.g. ``:=``).
        s = ReplSession()
        s.eval_entry("var x = 0")
        r = s.eval_entry("if true =>\n    x := 42\n| else =>\n    x := 0")
        assert r.ok
        assert r.kind == "expression"
        assert r.value == VOID_VALUE
        # The side effect was applied.
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["x"] == 42

    def test_if_expr_in_let_binding_echoes_value(self) -> None:
        # An if-expression used in a let binding produces a binding echo with
        # the correct value and type.
        s = ReplSession()
        r = s.eval_entry("let result = if true => 7 | else => 3")
        assert r.ok
        assert r.kind == "binding"
        assert r.name == "result"
        assert r.value is not None
        assert _int(r.value) == 7
        assert isinstance(r.value_type, IntType)


# ---------------------------------------------------------------------------
# Do-loop expression with assignment — covers _assign_targets_in_program Do branch
# ---------------------------------------------------------------------------


class TestDoExpr:
    def test_do_loop_assign_target_detected(self) -> None:
        # A ``do/until`` loop containing a ``:=`` mutation must be classified
        # as "expression" (not statement), and the ``:=`` side-effect must be
        # visible in the session after promotion.  This exercises the Do branch
        # in ``_assign_targets_in_program`` (session.py lines 80-81).
        s = ReplSession()
        s.eval_entry("var counter = 0")
        r = s.eval_entry("do\n  counter := counter + 1\nuntil counter >= 3\ncounter")
        assert r.ok
        assert r.kind == "expression"
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["counter"] == 3

    def test_do_loop_assignment_rolls_back_on_error(self) -> None:
        # A ``:=`` inside a failing do-loop entry rolls back atomically: the
        # var is restored to its pre-entry value.
        s = ReplSession()
        s.eval_entry("var x = 0")
        # The loop mutates x but the trailing type error kills the entry.
        r = s.eval_entry('do\n  x := x + 1\nuntil x >= 2\nlet bad: int = "oops"')
        assert not r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["x"] == 0  # rolled back


class TestIndexedAssignTargets:
    def test_nested_indexed_assign_rolls_back_on_error(self) -> None:
        s = ReplSession()
        s.eval_entry("var xs = [[1, 2]]")
        r = s.eval_entry('xs[0][1] := 9\nlet bad: int = "oops"')
        assert not r.ok
        vals = {n: v for n, _t, v in s.bindings()}
        assert vals["xs"].elements[0].elements[1] == IntValue(2)


# ---------------------------------------------------------------------------
# Try expression with assignment — covers _assign_targets_in_program Try branch
# ---------------------------------------------------------------------------


class TestTryExpr:
    def test_try_assign_target_detected_in_body(self) -> None:
        # A ``try`` expression containing a ``:=`` in its body must have the
        # ``:=`` target detected by ``_assign_targets_in_program``
        # so the var is included in atomic rollback tracking.
        s = ReplSession()
        s.eval_entry("var x = 0")
        r = s.eval_entry("try\n  x := 1\ncatch _ =>\n  x := 99\nx")
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["x"] == 1

    def test_try_assign_target_detected_in_handler(self) -> None:
        # A ``:=`` inside a catch handler must also be detected so
        # the var snapshot is captured before the entry runs.
        s = ReplSession()
        s.eval_entry("var x = 0")
        # The handler assignment path requires the try body to raise, which is tricky
        # to trigger without a real exception; we just verify that an assignment inside
        # try is promoted correctly (body succeeds, handler is not taken).
        r = s.eval_entry("try\n  x := 7\ncatch _ =>\n  x := 99\nx")
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["x"] == 7

    def test_try_assignment_rolls_back_on_type_error(self) -> None:
        # A type error in the same entry causes the whole entry to roll back,
        # including any ``:=`` in a try body.
        s = ReplSession()
        s.eval_entry("var x = 0")
        r = s.eval_entry('try\n  x := 5\ncatch _ =>\n  ()\nlet bad: int = "oops"')
        assert not r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["x"] == 0  # rolled back


# ---------------------------------------------------------------------------
# FuncDef (def) — declaration kind and cross-entry callability
# ---------------------------------------------------------------------------


class TestFuncDef:
    def test_funcdef_classified_as_declaration(self) -> None:
        # A bare ``def`` entry must be classified as "declaration" with the
        # function name as the declared name.
        s = ReplSession()
        r = s.eval_entry("def double(x: int) -> int = x * 2")
        assert r.ok
        assert r.kind == "declaration"
        assert r.name == "double"

    def test_funcdef_callable_in_subsequent_entry(self) -> None:
        # A function defined in one REPL entry must be callable in a later entry
        # (cross-entry callability via TypeEnvironment.seed_from + closure
        # promotion into session scope).
        s = ReplSession()
        s.eval_entry("def add(a: int, b: int) -> int = a + b")
        r = s.eval_entry("add(3, 4)")
        assert r.ok
        assert r.value is not None
        assert _int(r.value) == 7

    def test_typed_nullary_constructor_call_as_juxt_arg(self) -> None:
        s = ReplSession()
        r = s.eval_entry(
            "enum Opt[T]\n  | None\ndef f(x: Opt[int]) -> bool = false\nf Opt[int]::None()"
        )
        assert r.ok, r.diagnostics
        assert isinstance(r.value, BoolValue)
        assert r.value.value is False

    def test_funcdef_result_used_in_binding(self) -> None:
        # A function defined in entry 1 can be used in a let-binding in entry 2.
        s = ReplSession()
        s.eval_entry("def square(n: int) -> int = n * n")
        r = s.eval_entry("let result = square(5)")
        assert r.ok
        assert r.kind == "binding"
        assert r.name == "result"
        assert r.value is not None
        assert _int(r.value) == 25

    def test_funcdef_failed_entry_does_not_persist(self) -> None:
        # A function in a failing entry (type error) must not be callable in
        # the next entry — atomic rollback must erase the definition.
        s = ReplSession()
        bad = s.eval_entry('def broken(x: int) -> int = x\nlet y: int = "oops"')
        assert not bad.ok
        r = s.eval_entry("broken(1)")
        assert not r.ok  # broken not in scope

    def test_runtime_failure_promotes_scoped_types_by_completed_initializers(self) -> None:
        """A scoped type follows the same completion rule as a root type.

        The region completed before the failing statement, so its member stays
        usable; a region after the failure never took effect.
        """
        s = ReplSession()
        before = s.eval_entry("scope A\nrecord Token()\nend A\n1 / 0")
        assert not before.ok
        assert s.eval_entry("A::Token()").ok

        after = s.eval_entry("let z: decimal = 1 / 0\nscope B\nrecord Later()\nend B")
        assert not after.ok
        assert not s.eval_entry("B::Later()").ok

    def test_runtime_failure_promotes_only_the_completed_binder_in_a_region(self) -> None:
        """A region is not one promotable group: each binder completes on its own.

        ``A::a``'s initializer runs and completes before ``A::b``'s raises, so
        only ``a`` may be promoted — a failed sibling binder in the same region
        must not be treated as completed merely because it shares the region.
        """
        s = ReplSession()
        failed = s.eval_entry("scope A\nlet a = 1\nvar b = 1 / 0\nend A")
        assert not failed.ok
        assert s.eval_entry("A::a").value == IntValue(1)
        assert not s.eval_entry("A::b").ok

    def test_runtime_failure_after_region_form_scoped_let_gives_clean_diagnostic(self) -> None:
        """A region-form ``let`` past a failing sibling initializer never ran.

        Regression test: ``d``'s scope-member ref keys on its pattern binder
        candidate node id, not the ``LetDecl`` node's own id; the promotion
        plan's node-id set must be computed the same way, or ``d`` is treated
        as unconditionally promoted and later lookup crashes the session
        instead of diagnosing cleanly.
        """
        s = ReplSession()
        failed = s.eval_entry("scope A\nlet a = 1\nvar c = 1 / 0\nlet d = 4\nend A")
        assert not failed.ok
        r = s.eval_entry("A::d")
        assert not r.ok

    def test_runtime_failure_after_scoped_let_in_nested_region_gives_clean_diagnostic(
        self,
    ) -> None:
        """The region-form ``let`` fix also holds across nested regions."""
        s = ReplSession()
        failed = s.eval_entry(
            "scope A\nlet a = 1\nscope B\nvar c = 1 / 0\nlet d = 4\nend B\nlet e = 5\nend A"
        )
        assert not failed.ok
        assert not s.eval_entry("A::e").ok
        assert not s.eval_entry("A::B::d").ok

    def test_runtime_failure_after_scoped_let_in_repeated_region_gives_clean_diagnostic(
        self,
    ) -> None:
        """The region-form ``let`` fix also holds when the same scope is reopened."""
        s = ReplSession()
        failed = s.eval_entry("scope A\nlet a = 1\nend A\nscope A\nvar b = 1 / 0\nlet c = 4\nend A")
        assert not failed.ok
        assert not s.eval_entry("A::c").ok

    def test_runtime_failure_after_scoped_var_in_region_gives_clean_diagnostic(self) -> None:
        """Control: a ``var`` past a failing sibling was already handled correctly."""
        s = ReplSession()
        failed = s.eval_entry("scope A\nlet a = 1\nvar c = 1 / 0\nvar d = 4\nend A")
        assert not failed.ok
        assert not s.eval_entry("A::d").ok

    def test_runtime_failure_after_shorthand_scoped_let_gives_clean_diagnostic(self) -> None:
        """Control: the declaration-path shorthand form was already handled correctly."""
        s = ReplSession()
        failed = s.eval_entry("let A::a = 1\nvar A::c = 1 / 0\nlet A::d = 4")
        assert not failed.ok
        assert not s.eval_entry("A::d").ok


# ---------------------------------------------------------------------------
# REPL import support
# ---------------------------------------------------------------------------


class TestInfixDecl:
    """REPL persistence of user-defined infix operator declarations."""

    def test_infixl_usable_in_subsequent_entry(self) -> None:
        # ``infixl`` declared in one entry must make the operator usable in a
        # later entry (the fixity persists across entries for parsing).
        s = ReplSession()
        s.eval_entry("infixl +++ at 5")
        s.eval_entry("def +++(x: int, y: int) -> int = x + y")
        r = s.eval_entry("1 +++ 2")
        assert r.ok, r.diagnostics
        assert r.value is not None
        assert _int(r.value) == 3

    def test_infixr_usable_in_subsequent_entry(self) -> None:
        s = ReplSession()
        s.eval_entry("infixr << at 40")
        s.eval_entry('def <<(x: text, y: text) -> text = "(" + x + y + ")"')
        r = s.eval_entry('"a" << "b" << "c"')
        assert r.ok, r.diagnostics
        assert r.value is not None
        assert _text(r.value) == "(a(bc))"

    def test_infix_relative_priority_persists(self) -> None:
        # A relative priority (``at prio > + 1``) declared in one entry must
        # keep binding correctly when the operator is used in a later entry.
        s = ReplSession()
        s.eval_entry("infixl |> at prio > + 1")
        s.eval_entry("def |>(x: int, y: int) -> int = x * 10 + y")
        r = s.eval_entry("1 + 2 |> 3 > 20")
        assert r.ok, r.diagnostics
        assert r.value is not None
        # ((1+2) |> 3) > 20  =>  33 > 20  =>  true
        assert isinstance(r.value, BoolValue)
        assert r.value.value is True

    def test_infix_redefinition_shadows(self) -> None:
        # Redeclaring an infix operator in a later entry updates its fixity
        # (mirrors how ``let``/``record`` redefinitions shadow in the REPL).
        s = ReplSession()
        s.eval_entry("infixl +++ at 5")
        s.eval_entry("def +++(x: int, y: int) -> int = x + y")
        # Redeclare with a different priority; the operator is still usable.
        r_decl = s.eval_entry("infixl +++ at 7")
        assert r_decl.ok, r_decl.diagnostics
        r = s.eval_entry("1 +++ 2")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 3

    def test_infix_relative_priority_to_user_operator_persists(self) -> None:
        # A relative priority may reference a user operator declared in an earlier
        # entry; the reference resolves against the accumulated fixity.
        s = ReplSession()
        s.eval_entry("infixl +++ at 5")
        s.eval_entry("infixl *** at prio +++ + 1")
        s.eval_entry("def +++(x: int, y: int) -> int = x + y")
        s.eval_entry("def ***(x: int, y: int) -> int = x * y")
        # ``+++`` binds at 5, ``***`` at 6 (tighter), so ``1 +++ 2 *** 3``
        # groups as ``1 +++ (2 *** 3)`` = 1 + (2*3) = 7.
        r = s.eval_entry("1 +++ 2 *** 3")
        assert r.ok, r.diagnostics
        assert r.value is not None
        assert _int(r.value) == 7

    def test_infix_decl_survives_reset(self) -> None:
        # ``:reset`` clears ALL session state, including accumulated fixity.
        s = ReplSession()
        s.eval_entry("infixl +++ at 5")
        s.eval_entry("def +++(x: int, y: int) -> int = x + y")
        s.reset()
        r = s.eval_entry("1 +++ 2")
        assert not r.ok  # fixity gone after reset

    def test_type_of_uses_accumulated_infix(self) -> None:
        # ``:type`` parses with the session's accumulated fixity too.
        s = ReplSession()
        s.eval_entry("infixl +++ at 5")
        s.eval_entry("def +++(x: int, y: int) -> int = x + y")
        assert s.type_of("1 +++ 2") == "int"


class TestImports:
    """REPL import and use declaration support."""

    def _make_session_with_root(self, root: Path) -> ReplSession:
        """Create a ReplSession with *root* as the only module search root."""
        from agm.agl.modules.roots import assemble_roots

        roots = assemble_roots(
            invocation_root=root,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
            lib_root=None,
            configured=[],
            cli=[],
            cwd=root,
        )
        s = ReplSession()
        s._roots = roots  # inject roots directly
        return s

    def test_import_basic_function_call(self, tmp_path: Path) -> None:
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)
        # A wildcard import makes functions available unqualified.
        r = s.eval_entry("import mylib::*\nadd(3, 4)")
        assert r.ok, r.diagnostics
        assert r.kind == "expression"
        assert _int(r.value) == 7

    def test_imported_type_parameter_wildcard_is_rejected(self, tmp_path: Path) -> None:
        """A non-method '_' type parameter is rejected even from an imported module in the REPL."""
        (tmp_path / "generic.agl").write_text(
            "record Box[_, T]\n"
            "  value: T\n"
            "enum Result[_, T]\n"
            "  | ok(value: T)\n"
            "type Items[_, T] = array[T]\n"
        )
        s = self._make_session_with_root(tmp_path)

        result = s.eval_entry(
            "import generic\n"
            "let box: generic::Box[int] = generic::Box(value = 1)\n"
            'let result: generic::Result[text] = generic::Result::ok(value = "done")\n'
            "let items: generic::Items[int] = [box.value]\n"
            "items[0]"
        )

        assert not result.ok
        messages = " | ".join(d.message for d in result.diagnostics)
        assert "only allowed in a method's receiver type parameters" in messages
        assert "needs a name" in messages

    def test_import_persists_across_entries(self, tmp_path: Path) -> None:
        lib = tmp_path / "util.agl"
        lib.write_text("def double(x: int) -> int = x * 2\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("import util::*")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("double(5)")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 10

    def test_current_module_use_is_retained(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Source::value() -> int = 1").ok
        assert s.eval_entry("use ::Source::*").ok
        assert s.eval_entry("value()").value == IntValue(1)

    def test_same_target_use_is_replaced_and_failed_entries_preserve_it(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Source::original() -> int = 1").ok
        assert s.eval_entry("def Source::replacement() -> int = 2").ok
        assert s.eval_entry("use Source::{original}").ok
        assert s.eval_entry("original()").value == IntValue(1)

        assert s.eval_entry("use Source::{replacement}").ok
        assert not s.eval_entry("original()").ok
        assert s.eval_entry("replacement()").value == IntValue(2)

        failed = s.eval_entry('use Source::{original}\nlet bad: int = "not an int"')
        assert not failed.ok
        assert not s.eval_entry("original()").ok
        assert s.eval_entry("replacement()").value == IntValue(2)

    def test_single_member_alias_use_is_replaced_by_parent_target(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("use Source::old as selected").ok

        replacement = session.eval_entry("use Source::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("selected()").ok

    def test_single_member_alias_replacement_applies_to_its_entry_transactionally(
        self,
    ) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("use Source::{old}").ok

        failed = session.eval_entry("use Source::new as selected\nold()")

        assert not failed.ok
        assert session.eval_entry("old()").value == IntValue(1)

        replacement = session.eval_entry("use Source::new as selected\nselected()")

        assert replacement.ok, replacement.diagnostics
        assert replacement.value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_single_member_alias_replacement_can_reuse_the_exposed_name(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("use Source::old as selected").ok

        replacement = session.eval_entry("use Source::new as selected\nselected()")

        assert replacement.ok, replacement.diagnostics
        assert replacement.value == IntValue(2)

    def test_nested_whole_target_alias_does_not_replace_its_parent_use(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::Nested::value() -> int = 2").ok
        assert session.eval_entry("use Source::{old}").ok

        aliased = session.eval_entry("use Source::Nested as N\nold() + N::value()")

        assert aliased.ok, aliased.diagnostics
        assert aliased.value == IntValue(3)
        assert session.eval_entry("old()").value == IntValue(1)

    def test_nested_single_member_alias_replacement_keeps_other_retained_headers(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text("def value() -> int = 0\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry(
            "scope Outer\nimport lib\nscope Inner\nuse ::Source::{old}\nend Inner\nend Outer"
        ).ok

        replacement = session.eval_entry(
            "scope Outer\n"
            "scope Inner\n"
            "use ::Source::new as selected\n"
            "def read() -> int = selected()\n"
            "end Inner\n"
            "end Outer\n"
            "Outer::Inner::read()"
        )

        assert replacement.ok, replacement.diagnostics
        assert replacement.value == IntValue(2)
        assert not session.eval_entry(
            "scope Outer\nscope Inner\ndef stale() -> int = old()\nend Inner\nend Outer"
        ).ok

    def test_unrelated_import_alias_does_not_key_a_local_use(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text("def value() -> int = 0\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib as Other").ok
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("use Source::{old}").ok

        replacement = session.eval_entry("use ::Source::{new}")

        assert replacement.ok, replacement.diagnostics
        assert not session.eval_entry("old()").ok

    def test_nested_relative_and_current_module_use_targets_replace_each_other(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Outer::Source::old() -> int = 1").ok
        assert session.eval_entry("def Outer::Source::new() -> int = 2").ok
        assert session.eval_entry("scope Outer\nuse Source::{old}\nend Outer").ok

        replacement = session.eval_entry("scope Outer\nuse ::Outer::Source::{new}\nend Outer")

        assert replacement.ok, replacement.diagnostics
        assert not session.eval_entry(
            "scope Outer\ndef read() -> int = old()\nend Outer\nOuter::read()"
        ).ok

    def test_relative_use_does_not_replace_retained_target_after_nearer_scope_appears(
        self,
    ) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("scope Outer\nuse Source::{old}\nend Outer").ok
        assert session.eval_entry("def Outer::Source::new() -> int = 2").ok

        result = session.eval_entry(
            "scope Outer\n"
            "use Source::{new}\n"
            "def both() -> int = old() + new()\n"
            "end Outer\n"
            "Outer::both()"
        )

        assert result.ok, result.diagnostics
        assert result.value == IntValue(3)

    def test_replacing_nested_relative_use_hides_old_names_in_replacement_entry(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("scope Outer\nuse Source::{old}\nend Outer").ok

        replacement = session.eval_entry(
            "scope Outer\nuse Source::{new}\ndef captured() -> int = old()\nend Outer"
        )

        assert not replacement.ok
        assert not session.eval_entry("Outer::captured()").ok

    def test_regional_import_tail_does_not_canonicalize_an_unrelated_use(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            "scope Source\ndef old() -> int = 9\nend Source\n", encoding="utf-8"
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("def Right::Source::old() -> int = 1").ok
        assert session.eval_entry("def Right::Source::new() -> int = 2").ok
        assert session.eval_entry(
            "scope Left\nimport lib::{Source}\nend Left\nscope Right\nuse Source::{old}\nend Right"
        ).ok

        replacement = session.eval_entry("scope Right\nuse ::Right::Source::{new}\nend Right")

        assert replacement.ok, replacement.diagnostics
        assert not session.eval_entry(
            "scope Right\ndef read() -> int = old()\nend Right\nRight::read()"
        ).ok

    def test_glob_imported_scope_use_is_replaced_by_anchored_target(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text(
            "scope Source\ndef old() -> int = 1\ndef new() -> int = 2\nend Source\n",
            encoding="utf-8",
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib::*\nuse Source::{old}").ok

        replacement = session.eval_entry("use /lib::Source::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_wildcard_facade_use_survives_alias_replacement(self, tmp_path: Path) -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("def old() -> int = 1\n", encoding="utf-8")
        (package / "b.agl").write_text("def new() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import pkg/* as Old\nuse Old::{old}").ok

        replacement = session.eval_entry("import pkg/* as New\nuse New::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_anchored_use_replaces_suffix_use_of_same_module(self, tmp_path: Path) -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "lib.agl").write_text(
            "def old() -> int = 1\ndef new() -> int = 2\n", encoding="utf-8"
        )
        s = self._make_session_with_root(tmp_path)
        assert s.eval_entry("import pkg/lib\nuse lib::{old}").ok

        replacement = s.eval_entry("use /pkg/lib::{new}")

        assert replacement.ok, replacement.diagnostics
        assert s.eval_entry("new()").value == IntValue(2)
        assert not s.eval_entry("old()").ok

    def test_retained_local_use_ignores_a_later_colliding_import(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text("def imported() -> int = 2\n")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("def Source::local() -> int = 1").ok
        assert session.eval_entry("use Source::*").ok

        imported = session.eval_entry("import lib as Source")
        local = session.eval_entry("local()")

        assert imported.ok, imported.diagnostics
        assert local.ok, local.diagnostics
        assert local.value == IntValue(1)

    def test_retained_import_canonicalizes_a_later_use_replacement(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text(
            "def old() -> int = 1\ndef new() -> int = 2\n", encoding="utf-8"
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib").ok
        assert session.eval_entry("use lib::{old}").ok

        replacement = session.eval_entry("use /lib::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_retained_imported_use_survives_import_alias_change(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text("def value() -> int = 1\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib as Old").ok
        assert session.eval_entry("use Old::*").ok

        changed = session.eval_entry("import lib as New")
        result = session.eval_entry("value()")

        assert changed.ok, changed.diagnostics
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_retained_imported_use_reports_when_a_replacement_hides_its_target(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            "scope S\ndef value() -> int = 1\nend S\n", encoding="utf-8"
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib\nuse lib::S::*").ok

        replacement = session.eval_entry("let other = 2\nimport lib hiding S")

        assert not replacement.ok
        assert replacement.diagnostics[0].message
        # The complaint locates the import that hid the route, not a placeholder line.
        assert replacement.diagnostics[0].line == 2

    def test_retained_imported_use_keeps_nested_scope_routes_after_alias_change(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            "scope Nested\ndef value() -> int = 1\nend Nested\n", encoding="utf-8"
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib as Old").ok
        assert session.eval_entry("use Old::*").ok
        assert session.eval_entry("import lib as New").ok

        result = session.eval_entry("use Nested::*\nvalue()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_import_tail_rename_canonicalizes_use_replacement(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text(
            "scope Source\ndef old() -> int = 1\ndef new() -> int = 2\nend Source\n"
            "scope Other\ndef value() -> int = 3\nend Other\n",
            encoding="utf-8",
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib::{Source as Alias, Other}").ok
        assert session.eval_entry("use Alias::{old}").ok

        replacement = session.eval_entry("use /lib::Source::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_wildcard_alias_facade_use_is_retained(self, tmp_path: Path) -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        (package / "b.agl").write_text("def second() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)

        assert session.eval_entry("import pkg/* as Facade\nuse Facade::*").ok
        assert session.eval_entry("first() + second()").value == IntValue(3)

    def test_retained_wildcard_facade_use_survives_a_shrinking_import(self, tmp_path: Path) -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        removed = package / "b.agl"
        removed.write_text("def second() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import pkg/* as Facade\nuse Facade::*").ok
        removed.unlink()

        result = session.eval_entry("first()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)
        assert not session.eval_entry("second()").ok

    def test_retained_wildcard_facade_refreshes_only_its_original_import(
        self, tmp_path: Path
    ) -> None:
        package = tmp_path / "pkg"
        unrelated = tmp_path / "unrelated"
        package.mkdir()
        unrelated.mkdir()
        (package / "a.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        (unrelated / "c.agl").write_text("def intruder() -> int = 3\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)

        assert session.eval_entry("import pkg/* as Facade\nuse Facade::*").ok
        (package / "b.agl").write_text("def second() -> int = 2\n", encoding="utf-8")
        assert session.eval_entry("import unrelated/* as Facade").ok

        refreshed = session.eval_entry("first() + second()")
        assert refreshed.ok, refreshed.diagnostics
        assert refreshed.value == IntValue(3)
        assert not session.eval_entry("intruder()").ok

        assert session.eval_entry("import pkg/a as Direct").ok
        fallback = session.eval_entry("first() + second()")
        assert fallback.ok, fallback.diagnostics
        assert fallback.value == IntValue(3)

    def test_named_scope_retains_a_wildcard_facade_use_of_an_imported_enum_across_entries(
        self, tmp_path: Path
    ) -> None:
        """A ``use`` targeting an imported module inside a named ``scope``
        region -- not the REPL root -- is retained and replayed the same way
        a root-level retained use already is, including the enum
        constructors a wildcard selection exposes."""
        (tmp_path / "lib.agl").write_text("enum Color\n  | Red\n  | Blue\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import lib\nscope Outer\nuse lib::*\nend Outer").ok

        declared = session.eval_entry(
            "scope Outer\n"
            "def describe(c: Color) -> int = case c of | Color::Red => 1 | Color::Blue => 2\n"
            "end Outer"
        )
        assert declared.ok, declared.diagnostics
        result = session.eval_entry("Outer::describe(lib::Color::Red)")
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

    def test_named_scope_retained_wildcard_facade_use_survives_a_shrinking_import(
        self, tmp_path: Path
    ) -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        removed = package / "b.agl"
        removed.write_text("def second() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry(
            "import pkg/* as Facade\nscope Outer\nuse Facade::*\nend Outer"
        ).ok
        removed.unlink()

        declared = session.eval_entry("scope Outer\ndef readFirst() -> int = first()\nend Outer")
        assert declared.ok, declared.diagnostics
        result = session.eval_entry("Outer::readFirst()")
        assert result.ok, result.diagnostics
        assert result.value == IntValue(1)

        stale = session.eval_entry("scope Outer\ndef readSecond() -> int = second()\nend Outer")
        assert not stale.ok

    def test_named_scope_retained_wildcard_facade_use_survives_direct_reimport_of_its_members(
        self, tmp_path: Path
    ) -> None:
        """Once every module a wildcard facade discovered is ALSO reimported
        directly under its own alias, the facade's own resolved origin no
        longer matches any current declaration -- the retained contribution
        must fall back to what is still independently importable rather than
        losing its members."""
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        (package / "b.agl").write_text("def second() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry(
            "import pkg/* as Facade\nscope Outer\nuse Facade::*\nend Outer"
        ).ok

        assert session.eval_entry("import pkg/a as Q1\nimport pkg/b as Q2").ok

        declared = session.eval_entry(
            "scope Outer\n"
            "def readFirst() -> int = first()\n"
            "def readSecond() -> int = second()\n"
            "end Outer"
        )
        assert declared.ok, declared.diagnostics
        result = session.eval_entry("Outer::readFirst() + Outer::readSecond()")
        assert result.ok, result.diagnostics
        assert result.value == IntValue(3)

    def test_retained_deep_wildcard_facade_member_use_tolerates_a_removed_module(
        self, tmp_path: Path
    ) -> None:
        """A retained use naming one wildcard-discovered module directly (not
        through the facade alias) simply stops seeing that module once it is
        removed from disk, rather than rejecting the unrelated entry that
        triggers the replay."""
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "a.agl").write_text("scope S\ndef value() -> int = 1\nend S\n", encoding="utf-8")
        (package / "b.agl").write_text("def other() -> int = 2\n", encoding="utf-8")
        removed = package / "a.agl"
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry("import pkg/*\nuse /pkg/a::S::*").ok
        removed.unlink()

        result = session.eval_entry("1")

        assert result.ok, result.diagnostics

    def test_retained_imported_use_with_constructors_survives_a_narrower_replacement(
        self, tmp_path: Path
    ) -> None:
        """A retained imported wildcard use inside a named scope, that
        exposed record constructors alongside plain bindings from TWO
        overlapping modules -- each also named through both its bare and its
        ``/``-anchored spelling -- keeps its plain bindings reachable once a
        later entry narrows every one of those four declarations down to
        just them. Two overlapping contributions per module (bare + anchored
        spellings of the same target) exercise every shape the constructor
        and binding retraction can see: an atom already retracted by an
        earlier contribution, one a later contribution still shares, and one
        a contribution owns outright."""
        (tmp_path / "a.agl").write_text(
            "record Point\n  x: int\ndef onlyA() -> int = 11\n", encoding="utf-8"
        )
        (tmp_path / "b.agl").write_text(
            "record Point\n  y: int\ndef onlyB() -> int = 22\n", encoding="utf-8"
        )
        session = self._make_session_with_root(tmp_path)
        assert session.eval_entry(
            "import a\nimport b\nscope Outer\nuse a::*\nuse /a::*\nuse b::*\nuse /b::*\nend Outer"
        ).ok

        narrowed = session.eval_entry(
            "scope Outer\n"
            "use a::{onlyA}\nuse /a::{onlyA}\nuse b::{onlyB}\nuse /b::{onlyB}\nend Outer"
        )
        assert narrowed.ok, narrowed.diagnostics

        result = session.eval_entry(
            "scope Outer\ndef check() -> int = onlyA() + onlyB()\nend Outer"
        )
        assert result.ok, result.diagnostics
        assert session.eval_entry("Outer::check()").value == IntValue(33)

    def test_retained_separate_wildcard_aliases_do_not_form_one_facade(
        self, tmp_path: Path
    ) -> None:
        alpha = tmp_path / "alpha"
        beta = tmp_path / "beta"
        alpha.mkdir()
        beta.mkdir()
        (alpha / "one.agl").write_text("def first() -> int = 1\n", encoding="utf-8")
        (beta / "two.agl").write_text("def second() -> int = 2\n", encoding="utf-8")
        session = self._make_session_with_root(tmp_path)

        assert session.eval_entry("import alpha/* as Facade").ok
        assert session.eval_entry("import beta/* as Facade").ok

        assert not session.eval_entry("use Facade::*").ok

    def test_local_and_current_module_use_spellings_replace_each_other(self) -> None:
        session = ReplSession()
        assert session.eval_entry("def Source::old() -> int = 1").ok
        assert session.eval_entry("def Source::new() -> int = 2").ok
        assert session.eval_entry("use Source::{old}").ok

        replacement = session.eval_entry("use ::Source::{new}")

        assert replacement.ok, replacement.diagnostics
        assert session.eval_entry("new()").value == IntValue(2)
        assert not session.eval_entry("old()").ok

    def test_import_alias_replacement_replaces_use_of_same_module(self, tmp_path: Path) -> None:
        (tmp_path / "lib.agl").write_text(
            "def old() -> int = 1\ndef new() -> int = 2\n", encoding="utf-8"
        )
        s = self._make_session_with_root(tmp_path)
        assert s.eval_entry("import lib as L\nuse L::{old}").ok

        replacement = s.eval_entry("import lib as X\nuse X::{new}")

        assert replacement.ok, replacement.diagnostics
        assert s.eval_entry("new()").value == IntValue(2)
        assert not s.eval_entry("old()").ok

    def test_plain_import_route_use_is_replaced_when_import_gains_alias(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "lib.agl").write_text(
            "def old() -> int = 1\ndef new() -> int = 2\n", encoding="utf-8"
        )
        s = self._make_session_with_root(tmp_path)
        assert s.eval_entry("import lib\nuse lib::{old}").ok

        replacement = s.eval_entry("import lib as X\nuse X::{new}")

        assert replacement.ok, replacement.diagnostics
        assert s.eval_entry("new()").value == IntValue(2)
        assert not s.eval_entry("old()").ok

    def test_distinct_use_targets_with_clashing_bare_names_remain_ambiguous(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def First::value() -> int = 1").ok
        assert s.eval_entry("def Second::value() -> int = 2").ok
        assert s.eval_entry("use First::*").ok
        assert s.eval_entry("use Second::*").ok

        assert not s.eval_entry("value()").ok
        assert s.eval_entry("First::value()").value == IntValue(1)
        assert s.eval_entry("Second::value()").value == IntValue(2)

    def test_same_target_uses_in_different_regions_remain_independent(self) -> None:
        s = ReplSession()
        assert s.eval_entry("def Source::value() -> int = 3").ok
        assert s.eval_entry("scope Left\nuse Source::*\nend Left").ok
        assert s.eval_entry("scope Right\nuse Source::*\nend Right").ok

        left = s.eval_entry("scope Left\ndef read() -> int = value()\nend Left\nLeft::read()")
        right = s.eval_entry("scope Right\ndef read() -> int = value()\nend Right\nRight::read()")

        assert left.ok, left.diagnostics
        assert left.value == IntValue(3)
        assert right.ok, right.diagnostics
        assert right.value == IntValue(3)

    def test_import_selected_members(self, tmp_path: Path) -> None:
        lib = tmp_path / "funcs.agl"
        lib.write_text("def square(n: int) -> int = n * n\ndef cube(n: int) -> int = n * n * n\n")
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import funcs::{square}\nsquare(4)")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 16

    def test_import_as_qualifier(self, tmp_path: Path) -> None:
        lib = tmp_path / "math.agl"
        lib.write_text("def inc(n: int) -> int = n + 1\n")
        s = self._make_session_with_root(tmp_path)
        # 'as' alias creates qualifier, use :: for qualified access
        r = s.eval_entry("import math as m\nm::inc(9)")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 10

    def test_self_ref_colon_colon(self, tmp_path: Path) -> None:
        # ::name should resolve to a prior session binding in program context
        s = self._make_session_with_root(tmp_path)
        s.eval_entry("let x = 42")
        lib = tmp_path / "refs.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        r = s.eval_entry("import refs\n::x")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 42

    def test_self_qualifier_in_repl_returns_session_binding(self, tmp_path: Path) -> None:
        # Regression: ::x in the REPL (program context) must return the
        # session-level binding, not a same-named lexical param.
        # A dummy lib import is used to trigger program context so ::name resolves correctly.
        lib = tmp_path / "dummy.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("let x = 100")
        assert r1.ok, r1.diagnostics
        # Import forces program context; ::x must still resolve to x=100, not the param.
        r2 = s.eval_entry("import dummy\ndef shadow(x: int) -> int = ::x\nshadow(7)")
        assert r2.ok, r2.diagnostics
        from agm.agl.semantics.values import IntValue

        assert r2.value == IntValue(100), f"Expected 100, got {r2.value}"

    def test_graph_entry_type_body_can_reference_prior_repl_type(self, tmp_path: Path) -> None:
        lib = tmp_path / "dummy.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("record R\n  x: int")
        assert r1.ok, r1.diagnostics

        r2 = s.eval_entry("import dummy\nrecord Box\n  r: R\nBox(r = R(x = 1))")

        assert r2.ok, r2.diagnostics

    def test_graph_entry_function_signature_can_reference_prior_repl_type(
        self, tmp_path: Path
    ) -> None:
        lib = tmp_path / "dummy.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("record R\n  x: int")
        assert r1.ok, r1.diagnostics

        r2 = s.eval_entry("import dummy\ndef get_x(r: R) -> int = r.x\nget_x(R(x = 2))")

        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 2

    def test_import_error_rollback(self, tmp_path: Path) -> None:
        lib = tmp_path / "goodlib.agl"
        lib.write_text("def val() -> int = 99\n")
        s = self._make_session_with_root(tmp_path)
        s.eval_entry("let keep = 1")
        before = _snapshot(s)
        # Entry imports goodlib but has a type error; module should NOT be cached
        r = s.eval_entry('import goodlib\nlet bad: int = "oops"')
        assert not r.ok
        assert _snapshot(s) == before
        # goodlib should NOT have been added to loaded lib modules
        from agm.agl.modules.ids import ModuleId

        assert ModuleId(segments=("goodlib",)) not in s._loaded_lib_modules

    def test_import_not_found_error(self, tmp_path: Path) -> None:
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import nonexistent\n1")
        assert not r.ok
        assert r.diagnostics

    def test_no_roots_set_but_import_attempted(self) -> None:
        # With only the stdlib root, an unrelated import should fail gracefully.
        s = ReplSession()
        from agm.agl.modules.roots import RootSet

        s._roots = RootSet(roots=frozenset({Path(__file__).resolve().parents[1] / "stdlib"}))
        r = s.eval_entry("import something\n1")
        assert not r.ok
        assert r.diagnostics

    def test_reuse_cached_module(self, tmp_path: Path) -> None:
        lib = tmp_path / "cached.agl"
        lib.write_text('def greet() -> text = "hello"\n')
        s = self._make_session_with_root(tmp_path)
        from agm.agl.modules.ids import ModuleId

        cached_id = ModuleId(segments=("cached",))
        # Import once to cache it with its bare members.
        r1 = s.eval_entry("import cached::*\ngreet()")
        assert r1.ok, r1.diagnostics
        # Now cached_id should be in loaded_lib_modules
        assert cached_id in s._loaded_lib_modules
        # Import again in next entry (uses cached module; greet() still in scope)
        r2 = s.eval_entry("greet()")
        assert r2.ok, r2.diagnostics

    def test_reset_clears_imports(self, tmp_path: Path) -> None:
        lib = tmp_path / "temp.agl"
        lib.write_text("def f() -> int = 1\n")
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import temp::*\nf()")
        assert r.ok, r.diagnostics
        s.reset()
        assert not s._loaded_lib_modules
        assert not s._accumulated_imports

    def test_interpreter_initialization_failure_rolls_back_link_image(self, tmp_path: Path) -> None:
        """A failed setting bootstrap leaves no linked declarations but consumes node ids."""
        from agm.agl.lower import LinkImage

        std_dir = tmp_path / "std"
        std_dir.mkdir()
        copyfile(
            Path(__file__).resolve().parents[1] / "stdlib" / "std" / "core.agl",
            std_dir / "core.agl",
        )
        config = std_dir / "config.agl"
        config.write_text(
            "import std/core::{Option, Agent}\n"
            'builtin var default-agent: Agent = AgentCommand("runner")\n'
            'builtin var timeout: Option[text] = Some("not-a-timeout")\n',
            encoding="utf-8",
        )
        session = ReplSession(stdlib_root=tmp_path)

        failed = session.eval_entry("import std/config\nlet stale = 1")

        assert not failed.ok
        assert session._link_image == LinkImage()
        next_node_id = session._next_node_id
        assert next_node_id > 0

        config.write_text(
            "import std/core::{Option, Agent}\n"
            'builtin var default-agent: Agent = AgentCommand("runner")\n'
            'builtin var timeout: Option[text] = Some("2s")\n',
            encoding="utf-8",
        )
        succeeded = session.eval_entry("import std/config\nlet fresh = 2")

        assert succeeded.ok, succeeded.diagnostics
        assert {name for name, _typ, _value in session.bindings()} == {"fresh"}
        assert session._session_scope.bindings["fresh"].decl_node_id >= next_node_id

    def test_companion_import_failure_rolls_back_link_image(self, tmp_path: Path) -> None:
        """A rejected entry leaves no linked declarations, however it was rejected.

        A failing companion import rejects the entry after lowering has already
        allocated into the persistent image, exactly as a failing setting
        bootstrap does, so both discard the same complete link delta.
        """
        from agm.agl.lower import LinkImage

        (tmp_path / "broken.agl").write_text("extern def f() -> int\n")
        (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
        s = self._make_session_with_root(tmp_path)

        failed = s.eval_entry("import broken::*\ndef helper() -> int\n  7\nlet stale = helper()")

        assert not failed.ok
        assert s._link_image == LinkImage()

        succeeded = s.eval_entry("def helper() -> int\n  1\nhelper()")

        assert succeeded.ok, succeeded.diagnostics
        assert _int(succeeded.value) == 1

    def test_runtime_failure_retains_completed_module_link(self, tmp_path: Path) -> None:
        # A library that completed before the entry raised keeps its loaded AST
        # and linked identities together. Re-importing must reuse both rather
        # than either reinitializing it or lowering fresh declaration IDs.
        from agm.agl.modules.ids import ModuleId

        lib = tmp_path / "boom.agl"
        lib.write_text("def f() -> int = 42\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("import boom\nlet z: decimal = 1 / 0")
        assert not r1.ok
        boom_id = ModuleId(("boom",))
        assert boom_id in s._loaded_lib_modules
        assert boom_id in s._link_image._linked_modules

        r2 = s.eval_entry("import boom::*\nf()")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 42

    def test_runtime_failure_prunes_entry_function_depending_on_incomplete_module(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "unstable.agl").write_text(
            "param seed: int = [1, 2][9]\ndef get_seed() -> int = seed\n"
        )
        session = self._make_session_with_root(tmp_path)

        failed = session.eval_entry("import unstable::*\ndef retained() -> int = get_seed()")

        assert not failed.ok
        assert "retained" not in failed.installed
        assert not session.eval_entry("retained()").ok

    def test_scope_error_in_graph_mode(self, tmp_path: Path) -> None:
        # Declaring a reserved built-in name as an agent in program context
        # triggers AglScopeError during resolve_program.
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import mylib\nagent ask")
        assert not r.ok
        assert r.diagnostics

    def test_check_only_graph_mode(self, tmp_path: Path) -> None:
        # check_only=True in program context returns a check result without evaluating.
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import mylib::*\nadd(1, 2)", check_only=True)
        assert r.ok, r.diagnostics
        # check_only does not promote session state.
        assert s.bindings() == []

    def test_check_only_graph_mode_rejects_invalid_unreachable_import(self, tmp_path: Path) -> None:
        lib = tmp_path / "invalid.agl"
        lib.write_text("def dormant(x: bool) -> int =\n  case x of\n    | true => 1\n")
        s = self._make_session_with_root(tmp_path)

        r = s.eval_entry("import invalid\n()", check_only=True)

        assert not r.ok
        assert r.error is None
        assert r.warnings == []
        assert [
            (
                diagnostic.severity,
                Path(diagnostic.source_label).name if diagnostic.source_label is not None else None,
                diagnostic.line,
            )
            for diagnostic in r.diagnostics
        ] == [("error", "invalid.agl", 2)]
        assert s.bindings() == []

    def test_agl_raise_in_graph_mode(self, tmp_path: Path) -> None:
        # An AglRaise exception during program evaluation aborts the entry.
        lib = tmp_path / "mylib.agl"
        lib.write_text('def boom() -> int = raise Abort(message = "boom")\n')
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import mylib::*\nboom()")
        assert not r.ok
        assert r.error is not None

    def test_agl_raise_in_graph_mode_records_exception_trace(self, tmp_path: Path) -> None:
        import json

        lib = tmp_path / "mylib.agl"
        lib.write_text('def boom() -> int = raise Abort(message = "boom")\n')
        trace = tmp_path / "trace.jsonl"
        s = self._make_session_with_root(tmp_path)
        s._trace_path = trace

        r = s.eval_entry("import mylib::*\nboom()")

        assert not r.ok
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        kinds = [rec["kind"] for rec in records]
        assert "exception" in kinds
        assert kinds[-1] == "run_end"
        assert records[-1]["ok"] is False

    def test_wildcard_reimport_replaces_each_matched_module_by_identity(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "add.agl").write_text("def add() -> int = 1\n")
        (tmp_path / "tools" / "mul.agl").write_text("def mul() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import tools/*::*\nadd() + mul()").ok
        replacement = s.eval_entry("import tools/add as arithmetic\narithmetic::add()")

        assert replacement.ok, replacement.diagnostics
        assert not s.eval_entry("add()").ok
        preserved = s.eval_entry("mul()")
        assert preserved.ok, preserved.diagnostics
        assert _int(preserved.value) == 2

    def test_direct_imports_replaced_by_wildcard_for_each_matched_module(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "add.agl").write_text("def add() -> int = 1\n")
        (tmp_path / "tools" / "mul.agl").write_text("def mul() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry(
            "import tools/add::*\nimport tools/mul as old_mul\nadd() + old_mul::mul()"
        ).ok
        replacement = s.eval_entry(
            "import tools/* as all_tools\nall_tools::add() + all_tools::mul()"
        )

        assert replacement.ok, replacement.diagnostics
        assert _int(replacement.value) == 3
        assert not s.eval_entry("add()").ok
        assert not s.eval_entry("old_mul::mul()").ok
        assert s.eval_entry("all_tools::add()").ok
        assert s.eval_entry("all_tools::mul()").ok

    def test_same_entry_imports_union_while_replacing_prior_declarations(
        self, tmp_path: Path
    ) -> None:
        lib = tmp_path / "math.agl"
        lib.write_text("def add() -> int = 1\ndef mul() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import math::*\nadd() + mul()").ok
        result = s.eval_entry(
            "import math::{add}\nimport math as arithmetic\nadd() + arithmetic::mul()"
        )

        assert result.ok, result.diagnostics
        assert _int(result.value) == 3
        assert s.eval_entry("add()").ok
        assert s.eval_entry("arithmetic::mul()").ok
        assert not s.eval_entry("mul()").ok

    def test_replacement_removes_all_prior_declarations_for_one_module(
        self, tmp_path: Path
    ) -> None:
        lib = tmp_path / "math.agl"
        lib.write_text("def add() -> int = 1\ndef mul() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry(
            "import math::{add}\nimport math as arithmetic\nadd() + arithmetic::mul()"
        ).ok
        replacement = s.eval_entry("import math hiding add\nmath::mul()")

        assert replacement.ok, replacement.diagnostics
        assert not s.eval_entry("add()").ok
        assert not s.eval_entry("arithmetic::mul()").ok
        assert not s.eval_entry("math::add()").ok
        assert s.eval_entry("math::mul()").ok

    def test_replacement_removes_alias_selection_and_hiding_options(self, tmp_path: Path) -> None:
        lib = tmp_path / "api.agl"
        lib.write_text("def alpha() -> int = 1\ndef beta() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import api::*\nalpha() + beta()").ok
        assert s.eval_entry("import api as old_api\nold_api::alpha()").ok
        assert not s.eval_entry("alpha()").ok
        assert not s.eval_entry("beta()").ok

        assert s.eval_entry("import api::{beta}\nbeta()").ok
        assert not s.eval_entry("old_api::alpha()").ok
        assert s.eval_entry("beta()").ok

        assert s.eval_entry("import api hiding beta\napi::alpha()").ok
        assert not s.eval_entry("beta()").ok
        assert not s.eval_entry("api::beta()").ok
        assert s.eval_entry("api::alpha()").ok

    def test_replacement_updates_suffix_and_anchored_contributions(self, tmp_path: Path) -> None:
        (tmp_path / "left").mkdir()
        (tmp_path / "right").mkdir()
        (tmp_path / "left" / "config.agl").write_text("def shared() -> int = 1\n")
        (tmp_path / "right" / "config.agl").write_text("def shared() -> int = 3\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import left/config").ok
        assert s.eval_entry("import right/config").ok
        assert not s.eval_entry("config::shared()").ok
        assert s.eval_entry("/left/config::shared()").ok

        replacement = s.eval_entry("import left/config hiding shared\nconfig::shared()")

        assert replacement.ok, replacement.diagnostics
        assert _int(replacement.value) == 3
        assert not s.eval_entry("/left/config::shared()").ok
        assert s.eval_entry("/right/config::shared()").ok

    def test_ensure_roots_lazy_init(self, tmp_path: Path) -> None:
        # When a ReplSession is created with cwd= but no explicit _roots,
        # _ensure_roots() builds the root set lazily on first import.
        lib = tmp_path / "lazylib.agl"
        lib.write_text("def val() -> int = 42\n")
        s = ReplSession(cwd=tmp_path)
        r = s.eval_entry("import lazylib::*\nval()")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 42
        # Roots were assembled lazily.
        assert s._roots is not None

    def test_contract_error_in_graph_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A ValueError from materialize_contract during program execution
        # returns a failed entry with a "Contract error:" diagnostic.
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object, type_table: object = None) -> object:
            raise ValueError("bad contract")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)
        from agm.agl.runtime.codec import TextCodec

        class BadCodec(TextCodec):
            @property
            def name(self) -> str:
                return "bad"

        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)
        # The custom-format ask produces a pre-lower contract materialization error.
        s.register_codec(BadCodec())
        r = s.eval_entry(
            'import mylib\nlet helper = AgentCommand("helper")\n'
            'ask("hi", agent = helper, format = "bad")'
        )
        assert not r.ok
        assert any("Contract error" in d.message for d in r.diagnostics)

    def test_parse_error_in_imported_module_has_source_label(self, tmp_path: Path) -> None:
        # Regression: parse error in an imported module must surface
        # with source_label pointing to the module file, not a bare line-1 diagnostic
        # with no location information.
        lib = tmp_path / "badmod.agl"
        lib.write_text("def bad = !!!\n")  # syntax error
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import badmod")
        assert not r.ok
        assert len(r.diagnostics) >= 1
        # The diagnostic must carry source_label pointing to the module file.
        assert r.diagnostics[0].source_label is not None
        assert "badmod" in r.diagnostics[0].source_label

    def test_module_not_found_surfaces_clean_diagnostic(self, tmp_path: Path) -> None:
        # Regression: ModuleNotFound must surface as a proper diagnostic
        # (not a raw exception stringified at line 1 with no module name context).
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import nonexistent_module_xyz")
        assert not r.ok
        assert len(r.diagnostics) >= 1
        assert "nonexistent_module_xyz" in r.diagnostics[0].message

    def test_wildcard_and_plain_import_coexist(self, tmp_path: Path) -> None:
        # Regression: import foo/* in entry1 and import foo in entry2
        # must BOTH persist. The dedup key must include the wildcard flag so they
        # don't clobber each other.
        # Create a submodule foo.a and a top-level module foo.
        foo_dir = tmp_path / "foo"
        foo_dir.mkdir()
        (foo_dir / "a.agl").write_text("def val() -> int = 42\n")
        (tmp_path / "foo.agl").write_text("def top() -> int = 99\n")
        s = self._make_session_with_root(tmp_path)
        # Entry 1: wildcard import of foo.* (imports foo.a, brings val into scope)
        r1 = s.eval_entry("import foo/*::*\nval()")
        assert r1.ok, r1.diagnostics
        # Entry 2: plain import of foo (brings top() into scope)
        r2 = s.eval_entry("import foo::*\ntop()")
        assert r2.ok, r2.diagnostics
        # Entry 3: val() from foo.a must still resolve (wildcard import persists)
        r3 = s.eval_entry("val()")
        assert r3.ok, r3.diagnostics
        from agm.agl.semantics.values import IntValue

        assert r3.value == IntValue(42)

    def test_retained_wildcard_picks_up_a_module_added_later(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "add.agl").write_text("def add() -> int = 1\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import tools/*::*\nadd()").ok

        (tmp_path / "tools" / "mul.agl").write_text("def mul() -> int = 2\n")
        later = s.eval_entry("mul()")

        assert later.ok, later.diagnostics
        assert _int(later.value) == 2

    def test_retained_wildcard_does_not_undo_a_later_module_replacement(
        self, tmp_path: Path
    ) -> None:
        """Re-expansion must not resurrect a declaration a later entry replaced."""
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "add.agl").write_text("def add() -> int = 1\n")
        (tmp_path / "tools" / "mul.agl").write_text("def mul() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import tools/*::*\nadd() + mul()").ok
        assert s.eval_entry("import tools/add as arithmetic\narithmetic::add()").ok

        assert not s.eval_entry("add()").ok
        assert s.eval_entry("arithmetic::add()").ok
        assert s.eval_entry("mul()").ok

    def test_generic_graph_load_error_surfaces_as_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Covers the last-resort ``except Exception`` fallback in
        # ``_eval_entry``: a generic error from the graph loader
        # (not an AglSyntaxError or module error) must still surface as a
        # failed entry with a diagnostic rather than an uncaught exception.
        import agm.agl.modules.loader as loader_mod

        original_build = loader_mod.build_repl_graph

        def bad_build(*args: object, **kwargs: object) -> object:
            raise RuntimeError("unexpected loader failure")

        lib = tmp_path / "mylib.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        monkeypatch.setattr(loader_mod, "build_repl_graph", bad_build)
        r = s.eval_entry("import mylib\nnoop(1)")
        assert not r.ok
        assert len(r.diagnostics) >= 1
        assert "unexpected loader failure" in r.diagnostics[0].message
        monkeypatch.setattr(loader_mod, "build_repl_graph", original_build)

    def test_region_scoped_import_retains_its_qualifier_route_across_entries(
        self, tmp_path: Path
    ) -> None:
        """A scoped import's module-wide qualifier route persists like the root spelling."""
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry(
            "scope A\nimport mylib::*\ndef go() -> int = add(1, 2)\nend A\nA::go()"
        ).ok
        r = s.eval_entry("mylib::add(1, 2)")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 3

    def test_region_scoped_import_retains_its_bare_narrowing_across_entries(
        self, tmp_path: Path
    ) -> None:
        """A later entry's same-named region still sees the earlier entry's scoped import."""
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry(
            "scope A\nimport mylib::*\ndef go() -> int = add(1, 2)\nend A\nA::go()"
        ).ok
        r = s.eval_entry("scope A\ndef go2() -> int = add(3, 4)\nend A\nA::go2()")
        assert r.ok, r.diagnostics
        assert _int(r.value) == 7

    def test_later_region_scoped_import_replaces_the_prior_selection(self, tmp_path: Path) -> None:
        (tmp_path / "mylib.agl").write_text("def x() -> int = 1\ndef y() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("scope A\nimport mylib::{x}\nend A").ok
        replacement = s.eval_entry("scope A\nimport mylib::{y}\nend A")

        assert replacement.ok, replacement.diagnostics
        old_selection = s.eval_entry("scope A\ndef old() -> int = x()\nend A")
        assert not old_selection.ok
        new_selection = s.eval_entry("scope A\ndef new() -> int = y()\nend A\nA::new()")
        assert new_selection.ok, new_selection.diagnostics
        assert new_selection.value == IntValue(2)

    def test_region_scoped_import_still_does_not_leak_bare_names_to_the_root(
        self, tmp_path: Path
    ) -> None:
        """Retention must not widen a scoped import's bare reach beyond its own region."""
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("scope A\nimport mylib::*\nend A").ok
        r = s.eval_entry("add(1, 2)")
        assert not r.ok

    def test_plain_region_scoped_import_retains_its_qualifier_route_across_entries(
        self, tmp_path: Path
    ) -> None:
        """A plain region-scoped import also persists across entries.

        A later entry extending the same region must still resolve the
        qualifier the import established, without redeclaring it.
        """
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry(
            "scope A\nimport mylib\ndef go() -> int = mylib::add(1, 2)\nend A\nA::go()"
        ).ok
        r = s.eval_entry("scope A\ndef go2() -> int = mylib::add(3, 4)\nend A\nA::go2()")

        assert r.ok, r.diagnostics
        assert _int(r.value) == 7

    def test_root_import_after_retained_scope_region_is_not_rejected(self, tmp_path: Path) -> None:
        """A retained region's preamble ordering must not gate a later root import.

        Regression: the preamble used to place retained scope regions before
        this entry's own root ``import``, which the header rule ("import and
        export declarations must appear before any other declarations")
        then rejected -- even though the entry's import is legitimately at
        its own root.
        """
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        (tmp_path / "other.agl").write_text("def mul(a: int, b: int) -> int = a * b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("scope A\nimport mylib::*\nend A").ok

        r = s.eval_entry("import other\nother::mul(2, 3)")

        assert r.ok, r.diagnostics
        assert _int(r.value) == 6

    def test_root_import_after_retained_purely_local_scope_region_is_not_rejected(
        self, tmp_path: Path
    ) -> None:
        """Same header-ordering regression, but the retained region has no import at all."""
        (tmp_path / "other.agl").write_text("def val() -> int = 5\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("scope Src\ndef v() -> int = 1\nend Src\nscope T\nuse Src::*\nend T").ok

        r = s.eval_entry("import other\nother::val()")

        assert r.ok, r.diagnostics
        assert _int(r.value) == 5

    def test_root_wildcard_import_survives_a_later_region_scoped_import_of_the_same_module(
        self, tmp_path: Path
    ) -> None:
        """A region-scoped import must not revoke an earlier root import's bare names.

        Regression: retention keyed the replacement decision on module identity
        alone, so a region-scoped ``import mylib`` clobbered the generation
        recorded for the earlier ROOT ``import mylib::*``, silently removing
        its bare names from later entries.
        """
        (tmp_path / "mylib.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("import mylib::*\nadd(1, 2)").ok
        assert s.eval_entry("scope A\nimport mylib\nend A").ok

        r = s.eval_entry("add(1, 2)")

        assert r.ok, r.diagnostics
        assert _int(r.value) == 3


# ---------------------------------------------------------------------------
# Unpromoted nominal declarations: a declaration a partially failed entry
# never promotes drops out of name resolution -- the previous declaration
# (or its absence) stays in effect for every later entry exactly as before
# the failed entry. This holds uniformly for every declaration kind,
# ``builtin`` included: the link image's nominal state -- descriptors and
# the host-mint table alike -- is rebuilt from the shared type table on
# every lowering, and that table excludes a declaration it marked as never
# having taken effect (``TypeTable.orphan``) while retaining the one that
# survived.
# ---------------------------------------------------------------------------


class TestUnpromotedNominalDeclarationEffects:
    def test_runtime_failure_leaves_the_previous_record_declaration_in_effect(self) -> None:
        s = ReplSession()
        assert s.eval_entry("record R\n  x: int").ok

        failed = s.eval_entry("let z: decimal = 1 / 0\nrecord R\n  y: int")
        assert not failed.ok

        construct = s.eval_entry("let f = R\nlet v = f(1)\nv.x")
        match = s.eval_entry("case R(x = 2) of\n  | R(x) => x")

        assert construct.ok, construct.diagnostics
        assert construct.value == IntValue(1)
        assert match.ok, match.diagnostics
        assert match.value == IntValue(2)

    def test_runtime_failure_leaves_the_previous_enum_declaration_in_effect(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum Color\n  | Red\n  | Green").ok

        failed = s.eval_entry("let z: decimal = 1 / 0\nenum Color\n  | Blue")
        assert not failed.ok

        construct = s.eval_entry("Color::Red")
        match = s.eval_entry(
            "let green: Color = Color::Green\ncase green of\n  | Red => 0\n  | Green => 1"
        )
        stale_variant = s.eval_entry("Color::Blue")

        assert construct.ok, construct.diagnostics
        assert match.ok, match.diagnostics
        assert match.value == IntValue(1)
        assert not stale_variant.ok

    def test_failed_enum_with_inline_members_leaves_no_scope_or_member_behind(self) -> None:
        s = ReplSession()

        failed = s.eval_entry("let z: decimal = 1 / 0\nenum Tree\n  | Node(value: int)")
        assert not failed.ok

        member = s.eval_entry("Tree::Node(value = 1)")
        later = s.eval_entry("42")

        assert not member.ok
        assert later.ok, later.diagnostics
        assert later.value == IntValue(42)

    def test_failed_enum_with_inline_members_preserves_same_named_prior_state(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum Tree\n  | Leaf(value: int)").ok

        failed = s.eval_entry("let z: decimal = 1 / 0\nenum Tree\n  | Node(value: int)")
        assert not failed.ok

        retained_type = s.eval_entry("def leaf_value(value: Tree::Leaf) -> int = value.value")
        retained = s.eval_entry("Tree::Leaf(value = 1)")
        unpromoted = s.eval_entry("Tree::Node(value = 1)")

        assert retained_type.ok, retained_type.diagnostics
        assert retained.ok, retained.diagnostics
        assert not unpromoted.ok

    def test_runtime_failure_leaves_the_previous_exception_declaration_in_effect(self) -> None:
        s = ReplSession()
        assert s.eval_entry("exception E extends Exception\n  code: int").ok

        failed = s.eval_entry(
            "let z: decimal = 1 / 0\nexception E extends Exception\n  label: text"
        )
        assert not failed.ok

        caught = s.eval_entry('try raise E(message = "boom", code = 1) catch E as e => e.code')
        stale_field = s.eval_entry('try raise E(message = "boom", label = "x") catch E as e => 0')

        assert caught.ok, caught.diagnostics
        assert caught.value == IntValue(1)
        assert not stale_field.ok

    def test_runtime_failure_leaves_a_first_declaration_entirely_undeclared(self) -> None:
        """A declaration with no PREVIOUS declaration to fall back to is simply
        absent -- not resurrected in some intermediate state -- once its own
        entry fails before promoting it."""
        s = ReplSession()

        failed = s.eval_entry("let z: decimal = 1 / 0\nrecord R\n  x: int")
        assert not failed.ok

        use = s.eval_entry("R(x = 1)")

        assert not use.ok

    def test_runtime_failure_leaves_the_canonical_builtin_identity_in_effect(self) -> None:
        """A ``builtin`` declaration is unpromoted the same way as any other
        kind: proven here by a later host mint of ``RangeError`` (no earlier
        program declaration exists) still landing on the canonical identity
        and remaining catchable, exactly as an ordinary nominal falls back to
        whatever preceded the failed entry."""
        s = ReplSession(default_stdlib=False)

        failed = s.eval_entry(
            "let z: decimal = 1 / 0\n"
            "scope Failed\n"
            "builtin exception RangeError extends Exception()\n"
            "end Failed"
        )
        assert not failed.ok

        caught = s.eval_entry(
            "let step = 0\n"
            "try\n"
            "  for i in 1 to 5 by step do\n"
            "    ()\n"
            "  done\n"
            "catch RangeError as error =>\n"
            "  ()"
        )

        assert caught.ok, caught.diagnostics

    def test_runtime_failure_leaves_the_previous_builtin_declaration_in_effect(self) -> None:
        """The same declaration kind, but with an earlier one already in
        effect: an unpromoted, scoped redeclaration must leave THAT earlier
        declaration's own spelling live rather than the failed entry's
        scoped spelling.

        A later ``catch RangeError`` cannot itself observe which identity is
        live here: the standard-library exception namespace always reseeds a
        recognized name like ``RangeError`` fresh in every later entry (see
        ``TypeEnvironment.seed_from``'s ``BUILTIN_EXCEPTIONS`` exclusion), so
        a catch clause resolves to the canonical identity regardless of any
        program declaration, in every entry but the declaring one itself.
        What is actually in effect is which spelling a fresh host mint
        raises under (``DeclaredNominal.display_name``, baked into the
        raised value's ``display_name`` at the mint site) -- a later entry's
        uncaught error reports the identity the mint actually used, which is
        the previously-declared, unscoped ``"RangeError"`` here, since the
        host-mint table is rebuilt from the shared type table on every
        lowering and that table still carries the earlier declaration under
        its own identity.
        """
        s = ReplSession(default_stdlib=False)
        declared = s.eval_entry("builtin exception RangeError extends Exception()")
        assert declared.ok, declared.diagnostics

        failed = s.eval_entry(
            "let z: decimal = 1 / 0\n"
            "scope Failed\n"
            "builtin exception RangeError extends Exception()\n"
            "end Failed"
        )
        assert not failed.ok

        raised = s.eval_entry("let step = 0\nfor i in 1 to 5 by step do\n  ()\ndone")

        assert not raised.ok
        assert raised.error is not None
        assert raised.error.type_name == "RangeError"


# ---------------------------------------------------------------------------
# extern def (Python FFI) in the REPL
# ---------------------------------------------------------------------------


class TestExternRepl:
    """REPL-specific ``extern def`` (Python FFI) session semantics.

    Companion loading, boundary crossing, and the full conversion matrix are
    covered end to end elsewhere (``test_agl_extern_loading.py``,
    ``test_agl_extern_runtime.py``); this class covers what is specific to
    the incremental REPL session: direct-entry placement rejection, a
    companion importing exactly once across entries via the session-held
    registry, ``:reset`` discarding that registry, and extern failures
    surfacing as catchable ``ExternError`` without derailing the session.
    """

    def _make_session_with_root(self, root: Path) -> ReplSession:
        from agm.agl.modules.roots import assemble_roots

        roots = assemble_roots(
            invocation_root=root,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
            lib_root=None,
            configured=[],
            cli=[],
            cwd=root,
        )
        s = ReplSession()
        s._roots = roots
        return s

    def _write_extern_lib(self, root: Path, name: str, agl: str, py: str) -> None:
        (root / f"{name}.agl").write_text(agl)
        (root / f"{name}.py").write_text(py)

    # -- Placement: a direct REPL entry has no backing file -----------------

    def test_direct_entry_extern_def_rejected(self) -> None:
        s = ReplSession()
        r = s.eval_entry("extern def f(x: int) -> int")
        assert not r.ok
        assert r.diagnostics

    def test_session_usable_after_rejected_extern_entry(self) -> None:
        s = ReplSession()
        s.eval_entry("extern def f(x: int) -> int")
        r = s.eval_entry("let x = 1 + 1")
        assert r.ok
        assert _int({name: value for name, _typ, value in s.bindings()}["x"]) == 2

    # -- Importing an extern-bearing library module --------------------------

    def test_import_makes_extern_callable_in_a_later_entry(self, tmp_path: Path) -> None:
        self._write_extern_lib(
            tmp_path,
            "extlib",
            "extern def add_one(x: int) -> int\n",
            "def add_one(x):\n    return x + 1\n",
        )
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("import extlib::*")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("add_one(41)")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 42

    def test_extern_is_first_class_across_entries(self, tmp_path: Path) -> None:
        self._write_extern_lib(
            tmp_path,
            "extlib",
            "extern def add_one(x: int) -> int\n",
            "def add_one(x):\n    return x + 1\n",
        )
        s = self._make_session_with_root(tmp_path)
        s.eval_entry("import extlib::*")
        r1 = s.eval_entry("let g = add_one")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("g(9)")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 10

    def test_missing_companion_uses_loader_diagnostic(self, tmp_path: Path) -> None:
        (tmp_path / "extlib.agl").write_text("extern def add_one(x: int) -> int\n")
        s = self._make_session_with_root(tmp_path)

        result = s.eval_entry("import extlib")

        assert result.ok is False
        assert result.diagnostics
        assert "companion" in result.diagnostics[0].message.lower()
        assert result.diagnostics[0].line == 1

    def test_companion_import_failure_leaves_the_repl_entry_unsuccessful(
        self, tmp_path: Path
    ) -> None:
        self._write_extern_lib(
            tmp_path,
            "broken",
            "extern def f() -> int\n",
            "raise RuntimeError('boom')\n",
        )
        session = self._make_session_with_root(tmp_path)

        result = session.eval_entry("import broken::*\nf()")

        assert not result.ok
        assert result.diagnostics

    # -- One extern registry per session: a companion imports exactly once --

    def test_companion_imports_exactly_once_across_entries_and_imports(
        self, tmp_path: Path
    ) -> None:
        marker = tmp_path / "marker.txt"
        self._write_extern_lib(
            tmp_path,
            "counting",
            "extern def touch() -> int\n",
            f"open({str(marker)!r}, 'a').write('x')\ndef touch():\n    return 1\n",
        )
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("import counting::*\ntouch()")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("touch()")
        assert r2.ok, r2.diagnostics
        # Re-importing the same module in a later entry is a no-op import.
        r3 = s.eval_entry("import counting::*\ntouch()")
        assert r3.ok, r3.diagnostics
        assert marker.read_text() == "x"

    # -- :reset discards the session's extern registry -----------------------

    def test_reset_gives_the_session_a_fresh_extern_registry(self) -> None:
        # The settled semantics: ``:reset`` discards extern state like every
        # other session-scoped binding.  What matters is that the session's
        # registry itself is a new object afterward — NOT whether the
        # underlying Python module object happens to still exist somewhere in
        # the process (an implementation detail this test does not pin down).
        s = ReplSession()
        registry_before = s._runtime.host_environment().extern_registry
        s.reset()
        registry_after = s._runtime.host_environment().extern_registry
        assert registry_before is not registry_after

    def test_import_and_call_still_work_after_reset(self, tmp_path: Path) -> None:
        self._write_extern_lib(
            tmp_path,
            "extlib",
            "extern def add_one(x: int) -> int\n",
            "def add_one(x):\n    return x + 1\n",
        )
        s = self._make_session_with_root(tmp_path)
        roots = s._roots
        r1 = s.eval_entry("import extlib::*\nadd_one(1)")
        assert r1.ok, r1.diagnostics
        s.reset()
        s._roots = roots  # :reset clears roots too; a real host re-supplies them
        r2 = s.eval_entry("import extlib::*\nadd_one(1)")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 2

    # -- Extern failures surface as ExternError; the session stays usable ----

    def test_uncaught_extern_error_fails_the_entry_and_session_continues(
        self, tmp_path: Path
    ) -> None:
        self._write_extern_lib(
            tmp_path,
            "extlib",
            "extern def boom() -> int\n",
            "def boom():\n    raise ValueError('kaboom')\n",
        )
        s = self._make_session_with_root(tmp_path)
        s.eval_entry("import extlib::*")
        r = s.eval_entry("boom()")
        assert not r.ok
        assert r.error is not None
        assert r.error.type_name == "ExternError"
        # The session is still usable after an uncaught extern failure.
        r2 = s.eval_entry("let x = 1 + 1")
        assert r2.ok
        assert _int({name: value for name, _typ, value in s.bindings()}["x"]) == 2

    def test_extern_error_is_catchable_and_renders_in_the_repl(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_extern_lib(
            tmp_path,
            "extlib",
            "extern def boom() -> int\n",
            "def boom():\n    raise ValueError('kaboom')\n",
        )
        s = self._make_session_with_root(tmp_path)
        s.eval_entry("import extlib::*")
        r = s.eval_entry(
            "let r = try\n  boom()\ncatch ExternError as e =>\n  print(e.function)\n  -1\n"
        )
        assert r.ok, r.diagnostics
        assert _int({name: value for name, _typ, value in s.bindings()}["r"]) == -1
        assert capsys.readouterr().out.strip() == "boom"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _int(value: object) -> int:
    from agm.agl.semantics.values import IntValue

    assert isinstance(value, IntValue)
    return value.value


def _text(value: object) -> str:
    from agm.agl.semantics.values import TextValue

    assert isinstance(value, TextValue)
    return value.value


def _snapshot(s: ReplSession) -> list[tuple[str, str, str]]:
    """A comparable snapshot of promoted bindings (name, type repr, value repr)."""
    return [(n, repr(t), repr(v)) for n, t, v in s.bindings()]


# ---------------------------------------------------------------------------
# has_runnable_statements — lexer-error defensive branch (Fix 2)
# ---------------------------------------------------------------------------


class TestHasRunnableStatements:
    def test_lexer_error_is_treated_as_runnable(self) -> None:
        """An unlexable entry must return True (treated as runnable).

        ``has_runnable_statements`` catches any lexer exception in the defensive
        ``except Exception`` arm and returns ``True`` so the entry flows to
        ``eval_entry`` and surfaces a real diagnostic rather than being silently
        dropped.  Verifying with ``'@'`` (which raises ``LexError``).
        """
        from agm.agl.repl.session import has_runnable_statements

        assert has_runnable_statements("@") is True
        assert has_runnable_statements('"unterminated') is True


# ---------------------------------------------------------------------------
# Closure / AgentValue REPL echo (Fix 1)
# ---------------------------------------------------------------------------


class TestFunctionAgentValueEcho:
    """Bare function and agent values at the prompt produce human-readable echo.

    Entering a bare name that resolves to a Closure (from a ``def`` or ``fn``
    expression) or an AgentValue must render a surface form — not crash the REPL.
    This tests the REPL echo path end-to-end via ``ReplSession.eval_entry``.
    """

    def test_bare_lambda_echo_does_not_crash(self) -> None:
        """A bare lambda expression echoes its surface form without crashing."""
        s = ReplSession()
        r = s.eval_entry("fn(x: int) -> int => x + 1")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        # The value is a Closure; render_value must not raise.
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        assert isinstance(r.value, IrClosureValue)
        rendered = render_value(r.value)
        assert rendered == "<function: int -> int>"

    def test_bare_def_name_echo_does_not_crash(self) -> None:
        """A bare function-name entry after a ``def`` echoes the surface form."""
        s = ReplSession()
        s.eval_entry("def dbl(x: int) -> int = x * 2")
        # Evaluating bare ``dbl`` returns the Closure.
        r = s.eval_entry("dbl")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        assert isinstance(r.value, IrClosureValue)
        rendered = render_value(r.value)
        assert rendered == "<function: int -> int>"

    def test_bindings_after_def_does_not_crash(self) -> None:
        """:bindings() after a ``def`` must not crash (Closure has a surface form)."""
        s = ReplSession()
        s.eval_entry("def dbl(x: int) -> int = x * 2")
        # bindings() returns Closure values; the meta-command renders them.
        binds = s.bindings()
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        assert any(isinstance(v, IrClosureValue) for _n, _t, v in binds)
        # render_value on each must not raise.
        for _n, _t, v in binds:
            render_value(v)  # must not raise TypeError


# ---------------------------------------------------------------------------
# Bare type-expression entries (REPL-only: print the type, don't error)
# ---------------------------------------------------------------------------


class TestBareTypeEntry:
    """A bare type expression at the REPL prints as a type instead of erroring.

    Typing a type (``int``, a declared enum/record name, a parameterized type)
    is not a value expression and previously surfaced ``'X' is not defined.``
    The REPL now recognizes such entries and echoes the resolved type.  This is
    a REPL-only convenience: the language, parser, and checker are unchanged.
    Entries that successfully evaluate as values are never intercepted.
    """

    def test_builtin_primitive_type_echoes_as_type(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        r = s.eval_entry("int")
        assert r.ok
        assert r.kind == "type"
        assert r.value is None
        assert isinstance(r.value_type, IntType)
        assert render_entry_result(r, echo=True) == "<type: int>"

    def test_builtin_container_types_echo_as_type(self) -> None:
        from agm.agl.semantics.types import ArrayType, DictType

        s = ReplSession()
        r = s.eval_entry("array[int]")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, ArrayType)
        assert isinstance(r.value_type.elem, IntType)

        r2 = s.eval_entry("dict[text, int]")
        assert r2.ok
        assert r2.kind == "type"
        assert isinstance(r2.value_type, DictType)

    def test_function_type_echoes_as_type(self) -> None:
        from agm.agl.semantics.types import FunctionType

        s = ReplSession()
        r = s.eval_entry("(int) -> bool")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, FunctionType)

    def test_declared_enum_name_echoes_as_type(self) -> None:
        from agm.agl.repl.render import render_entry_result
        from agm.agl.semantics.types import EnumType

        s = ReplSession()
        s.eval_entry("enum Color = Red | Green | Blue")
        r = s.eval_entry("Color")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, EnumType)
        assert (
            render_entry_result(r, echo=True)
            == "<type:\nenum Color\n  | Red\n  | Green\n  | Blue\n>"
        )

    def test_generic_type_application_echoes_as_type(self) -> None:
        from agm.agl.semantics.types import ArrayType

        s = ReplSession()
        s.eval_entry("type Pair[A, B] = array[A]")
        # The alias resolves transparently to its target: array[int].
        r = s.eval_entry("Pair[int, text]")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, ArrayType)
        assert isinstance(r.value_type.elem, IntType)

    def test_bare_generic_enum_name_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        s.eval_entry("enum Option[T]\n  | none\n  | some(value: T)")
        r = s.eval_entry("Option")
        assert r.ok
        assert r.kind == "type"
        assert r.value is None
        assert (
            render_entry_result(r, echo=True)
            == "<type:\nenum Option[T]\n  | none\n  | some(value: T)\n>"
        )

    def test_bare_generic_record_name_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        s.eval_entry("record Box[T]\n  value: T")
        r = s.eval_entry("Box")
        assert r.ok
        assert r.kind == "type"
        assert r.value is None
        assert render_entry_result(r, echo=True) == "<type:\nrecord Box[T]\n  value: T\n>"

    def test_use_exposed_generic_record_name_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        session = ReplSession()
        assert session.eval_entry("scope S\nrecord Box[T](value: T)\nend S").ok
        assert session.eval_entry("use S::*").ok

        result = session.eval_entry("Box")

        assert result.ok, result.diagnostics
        assert render_entry_result(result, echo=True) == "<type:\nrecord Box[T]\n  value: T\n>"

    @pytest.mark.parametrize(
        ("use_decl", "query", "display_name"),
        (
            ("use S::{Box}", "Box", "Box"),
            ("use S::{Box as Renamed}", "Renamed", "Renamed"),
        ),
    )
    def test_selective_use_exposed_generic_record_name_echoes_definition(
        self, use_decl: str, query: str, display_name: str
    ) -> None:
        from agm.agl.repl.render import render_entry_result

        session = ReplSession()
        assert session.eval_entry("scope S\nrecord Box[T](value: T)\nend S").ok
        assert session.eval_entry(use_decl).ok

        result = session.eval_entry(query)

        assert result.ok, result.diagnostics
        assert (
            render_entry_result(result, echo=True)
            == f"<type:\nrecord {display_name}[T]\n  value: T\n>"
        )

    def test_hidden_use_generic_record_name_does_not_echo_definition(self) -> None:
        session = ReplSession()
        assert session.eval_entry("scope S\nrecord Box[T](value: T)\nend S").ok
        assert session.eval_entry("use S::* hiding Box").ok

        assert not session.eval_entry("Box").ok

    def test_use_alias_qualified_generic_record_name_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        session = ReplSession()
        assert session.eval_entry("scope S\nrecord Box[T](value: T)\nend S").ok
        assert session.eval_entry("use S as Alias").ok

        result = session.eval_entry("Alias::Box")

        assert result.ok, result.diagnostics
        assert (
            render_entry_result(result, echo=True) == "<type:\nrecord Alias::Box[T]\n  value: T\n>"
        )

    def test_use_alias_does_not_expose_another_qualifier(self) -> None:
        session = ReplSession()
        assert session.eval_entry("scope S\nrecord Box[T](value: T)\nend S").ok
        assert session.eval_entry("use S as Alias").ok

        assert not session.eval_entry("Other::Box").ok

    @pytest.mark.parametrize(
        "local_route",
        (
            "use S as a\nscope S\nrecord Box[T](value: T)\nend S",
            "scope a\nrecord Box[T](value: T)\nend a",
        ),
    )
    def test_qualified_unapplied_generic_rejects_distinct_local_and_import_routes(
        self, tmp_path: Path, local_route: str
    ) -> None:
        (tmp_path / "a.agl").write_text("record Box[T](value: T)\n")
        session = ReplSession(
            cwd=tmp_path,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        )
        setup = session.eval_entry(f"import a\n{local_route}")
        assert setup.ok, setup.diagnostics

        assert not session.eval_entry("a::Box[int]").ok
        assert not session.eval_entry("a::Box").ok

    def test_qualified_unapplied_generic_displays_equivalent_duplicate_routes(
        self, tmp_path: Path
    ) -> None:
        from agm.agl.repl.render import render_entry_result

        (tmp_path / "a.agl").write_text("record Box[T](value: T)\n")
        session = ReplSession(
            cwd=tmp_path,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        )
        setup = session.eval_entry("import a\nuse /a as a")
        assert setup.ok, setup.diagnostics

        result = session.eval_entry("a::Box")

        assert result.ok, result.diagnostics
        assert render_entry_result(result, echo=True) == "<type:\nrecord a::Box[T]\n  value: T\n>"

    def test_use_alias_nested_generic_record_name_echoes_definition(self) -> None:
        session = ReplSession()
        assert session.eval_entry(
            "scope S\nscope Nested\nrecord Box[T](value: T)\nend Nested\nend S"
        ).ok
        assert session.eval_entry("use S as Alias").ok

        result = session.eval_entry("Alias::Nested::Box")

        assert result.ok, result.diagnostics
        assert result.kind == "type"

    def test_use_alias_non_generic_enum_falls_back_to_type_display(self) -> None:
        session = ReplSession()
        assert session.eval_entry("scope S\nenum Status | ready\nend S").ok
        assert session.eval_entry("use S as Alias").ok

        result = session.eval_entry("Alias::Status")

        assert result.ok, result.diagnostics
        assert result.kind == "type"

    def test_bare_scoped_generic_record_name_echoes_definition(self) -> None:
        """A retained named-scope generic resolves by its qualified local name.

        ``A::Box``'s qualifier names a local ``scope A``, not an import route;
        the bare qualified name must resolve the same way the unqualified
        root form does, rather than being treated as an (absent) import.
        """
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        s.eval_entry("scope A\nrecord Box[T]\n  value: T\nend A")
        r = s.eval_entry("A::Box")
        assert r.ok
        assert r.kind == "type"
        assert r.value is None
        assert render_entry_result(r, echo=True) == "<type:\nrecord A::Box[T]\n  value: T\n>"

    def test_bare_generic_type_entry_in_check_only_mode(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        s.eval_entry("enum Option[T]\n  | none\n  | some(value: T)")
        r = s.eval_entry("Option", check_only=True)
        assert r.ok
        assert r.kind == "type"
        assert (
            render_entry_result(r, echo=True, check_only=True)
            == "<type:\nenum Option[T]\n  | none\n  | some(value: T)\n>"
        )

    def test_implicit_core_import_bare_generic_type_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")
        r = s.eval_entry("Option")
        assert r.ok
        assert r.kind == "type"
        assert r.value is None
        assert (
            render_entry_result(r, echo=True)
            == "<type:\nenum Option[T]\n  | None\n  | Some(value: T)\n>"
        )

    def test_implicit_core_import_qualified_generic_type_echoes_definition(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")
        r = s.eval_entry("std/core::Option")
        assert r.ok
        assert r.kind == "type"
        assert (
            render_entry_result(r, echo=True)
            == "<type:\nenum std/core::Option[T]\n  | None\n  | Some(value: T)\n>"
        )

    def test_builtin_type_entry_works_when_graph_env_is_unavailable(self, tmp_path: Path) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession(stdlib_root=tmp_path / "missing-stdlib")
        r = s.eval_entry("int")
        assert r.ok
        assert r.kind == "type"
        assert render_entry_result(r, echo=True) == "<type: int>"

    def test_ambiguous_imported_generic_entry_keeps_original_failure(self) -> None:
        from agm.agl.modules.ids import ModuleId
        from agm.agl.parser import parse_type_expr
        from agm.agl.scope.imports import ImportEnv
        from agm.agl.semantics.types import RecordType
        from agm.agl.typecheck.env import GenericTypeDef, TypeEnvironment

        left = ModuleId(("left",))
        right = ModuleId(("right",))
        left_def = GenericTypeDef(
            kind="record",
            type_params=("T",),
            template=RecordType(name="Box", module_id=left),
        )
        right_def = GenericTypeDef(
            kind="record",
            type_params=("T",),
            template=RecordType(name="Box", module_id=right),
        )
        env = TypeEnvironment(
            program_generic_table={(left, "Box"): left_def, (right, "Box"): right_def},
            import_env=ImportEnv(
                contributions={},
                unqualified={"Box": frozenset({(left, "Box"), (right, "Box")})},
            ),
        )
        s = ReplSession()
        assert s._try_generic_type_entry(parse_type_expr("Box"), env) is None

    def test_qualified_unapplied_generic_resolution_edges(self) -> None:
        from agm.agl.modules.ids import ENTRY_ID, ModuleId
        from agm.agl.parser import parse_type_expr
        from agm.agl.scope.imports import ImportEnv, ModuleContribution
        from agm.agl.semantics.types import RecordType
        from agm.agl.syntax.types import NameT
        from agm.agl.typecheck.env import GenericTypeDef, TypeEnvironment

        local_def = GenericTypeDef(
            kind="record",
            type_params=("T",),
            template=RecordType(name="Box"),
        )
        local_expr = parse_type_expr("::Box")
        assert isinstance(local_expr, NameT)
        assert local_expr.qualifier is not None

        local_env = TypeEnvironment()
        local_env.register_generic_type("Box", local_def)
        assert local_env.resolve_qualified_unapplied_generic_type(
            local_expr.qualifier,
            "Box",
        ) == ("Box", local_def)

        empty_env = TypeEnvironment()
        assert (
            empty_env.resolve_qualified_unapplied_generic_type(
                local_expr.qualifier,
                "Box",
            )
            is None
        )

        graph_env = TypeEnvironment(program_generic_table={(ENTRY_ID, "Box"): local_def})
        assert graph_env.resolve_qualified_unapplied_generic_type(
            local_expr.qualifier,
            "Box",
        ) == ("Box", local_def)

        lib = ModuleId(("lib",))
        qualified_expr = parse_type_expr("missing::Box")
        assert isinstance(qualified_expr, NameT)
        assert qualified_expr.qualifier is not None
        no_handle_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(contributions={}, unqualified={}),
        )
        assert (
            no_handle_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.qualifier,
                "Box",
            )
            is None
        )

        no_name_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(
                contributions={lib: ModuleContribution(lib, {}, False, frozenset({"missing"}))},
                unqualified={},
            ),
        )
        assert (
            no_name_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.qualifier,
                "Box",
            )
            is None
        )

        missing = ModuleId(("missing",))
        no_generic_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(
                contributions={
                    missing: ModuleContribution(
                        missing,
                        {"Box": (missing, "Box")},
                        True,
                        frozenset(),
                        {"Box": (missing, "Box")},
                    )
                },
                unqualified={},
            ),
        )
        assert (
            no_generic_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.qualifier,
                "Box",
            )
            is None
        )

    def test_qualified_unapplied_generic_resolution_prefers_local_named_scope(self) -> None:
        """A non-empty qualifier may name a local scope, not only an import route.

        ``A::Box`` for a locally-declared ``scope A`` generic must resolve
        without any import environment; a ``/``-anchored qualifier (always an
        import route) must never be resolved against the same local name.
        """
        from agm.agl.parser import parse_type_expr
        from agm.agl.semantics.types import RecordType
        from agm.agl.syntax.types import NameT
        from agm.agl.typecheck.env import GenericTypeDef, TypeEnvironment

        local_def = GenericTypeDef(
            kind="record",
            type_params=("T",),
            template=RecordType(name="Box", scope_path=("A",)),
        )
        env = TypeEnvironment()
        env.register_generic_type("A::Box", local_def)

        local_expr = parse_type_expr("A::Box")
        assert isinstance(local_expr, NameT)
        assert local_expr.qualifier is not None
        assert env.resolve_qualified_unapplied_generic_type(
            local_expr.qualifier,
            "Box",
        ) == ("A::Box", local_def)

        module_expr = parse_type_expr("/A::Box")
        assert isinstance(module_expr, NameT)
        assert module_expr.qualifier is not None
        assert (
            env.resolve_qualified_unapplied_generic_type(
                module_expr.qualifier,
                "Box",
            )
            is None
        )

    def test_record_name_still_evaluates_as_constructor(self) -> None:
        # A record name doubles as a constructor value, so it must keep
        # evaluating normally (the type fallback only triggers on failure).
        from agm.agl.semantics.values import ConstructorValue

        s = ReplSession()
        s.eval_entry("record Point(x: int, y: int)")
        r = s.eval_entry("Point")
        assert r.ok
        assert r.kind == "expression"
        assert isinstance(r.value, ConstructorValue)

    def test_binding_name_not_intercepted_as_type(self) -> None:
        # ``x`` parses as a type expression (a NameT), but it is a live value
        # binding that evaluates successfully, so it must NOT be intercepted.
        s = ReplSession()
        s.eval_entry("let x = 5")
        r = s.eval_entry("x")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 5

    def test_expression_not_intercepted_as_type(self) -> None:
        # ``1 + 2`` does not parse as a type expression; it evaluates normally.
        s = ReplSession()
        r = s.eval_entry("1 + 2")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        assert _int(r.value) == 3

    def test_truly_undefined_name_keeps_original_error(self) -> None:
        # ``nope`` parses as a type expression but does not resolve to a known
        # type, so the original "is not defined" error is preserved.
        s = ReplSession()
        r = s.eval_entry("nope")
        assert not r.ok
        assert r.kind != "type"
        assert any("not defined" in d.message for d in r.diagnostics)

    def test_type_entry_does_not_mutate_session_state(self) -> None:
        # Like ``:type``, a bare type entry must not promote, advance node ids,
        # or install any binding.
        s = ReplSession()
        before = s._next_node_id
        s.eval_entry("int")
        assert s._next_node_id == before
        assert s.bindings() == []
        assert s.type_names() == frozenset()

    def test_type_entry_echo_respects_echo_off(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        r = s.eval_entry("int")
        assert r.ok
        assert render_entry_result(r, echo=False) is None

    def test_type_entry_in_check_only_mode(self) -> None:
        from agm.agl.repl.render import render_entry_result

        s = ReplSession()
        r = s.eval_entry("int", check_only=True)
        assert r.ok
        assert r.kind == "type"
        assert render_entry_result(r, echo=True, check_only=True) == "<type: int>"


# ---------------------------------------------------------------------------
# Local `use` narrowing across REPL entries
# ---------------------------------------------------------------------------


class TestLocalUseNarrowing:
    """A retained local-scope ``use`` stays in sync with later narrowing ``use``s.

    Mirrors how a region-scoped ``import`` narrowing replaces the prior
    selection (see ``TestImports``): a later, narrower ``use`` of the same
    local scope must supersede an earlier glob ``use``, not merely add to it.
    """

    def test_narrowing_local_use_retracts_a_member_exposed_by_an_earlier_glob_use(self) -> None:
        session = ReplSession()
        assert session.eval_entry("scope S\ndef foo() -> int = 1\ndef bar() -> int = 2\nend S").ok
        assert session.eval_entry("use S::*").ok
        assert session.eval_entry("use S::{bar}").ok

        assert session.eval_entry("bar()").ok
        assert not session.eval_entry("foo()").ok

    def test_narrowing_local_use_in_a_named_scope_retracts_a_type_in_a_later_entry(
        self,
    ) -> None:
        """A tail-selecting (non-glob) local use, retained on a NAMED scope,

        must stop exposing its selected type once a later ``use`` on the same
        target supersedes it -- even in an entry that declares no ``use`` of
        its own, and even though type resolution has no live fallback and
        depends entirely on the retained static bare-contribution table.
        """
        session = ReplSession()
        assert session.eval_entry("type Source::Meters = int").ok
        assert session.eval_entry("type Source::Seconds = int").ok
        assert session.eval_entry("scope Outer\nuse Source::{Meters}\nend Outer").ok
        assert session.eval_entry("scope Outer\nuse Source::{Seconds}\nend Outer").ok

        result = session.eval_entry("scope Outer\nlet duration: Seconds = 1\nend Outer")
        assert result.ok, result.diagnostics

        assert not session.eval_entry("scope Outer\nlet distance: Meters = 1\nend Outer").ok

    def test_a_narrower_use_supersedes_a_retained_local_scope_route_within_the_same_entry(
        self,
    ) -> None:
        """A local scope reached only through an earlier bare ``use`` (not
        declared as a top-level scope of its own) stops being nameable once
        that earlier ``use`` is narrowed away by a later declaration in the
        SAME entry -- the earlier declaration is retained from a prior entry,
        so the narrowing here must see it as superseded mid-entry, not only
        once the next entry rebuilds the scope from scratch."""
        session = ReplSession()
        assert session.eval_entry(
            "scope A\nscope B\ndef value() -> int = 1\nend B\ndef direct() -> int = 5\nend A"
        ).ok
        assert session.eval_entry("use A::*").ok

        narrowed = session.eval_entry("use A::{direct}\nuse B::*")

        assert not narrowed.ok

    def test_narrowing_local_use_retracts_constructors_shared_by_overlapping_glob_uses(
        self,
    ) -> None:
        """Mirrors ``test_narrowing_local_use_retracts_a_member_exposed_by_an_earlier_glob_use``
        for the record constructors two overlapping local wildcard ``use``s
        inside a named scope expose, not just their plain bindings: a bare
        constructor pattern needs the source scopes' own constructor
        candidates, so it stops matching once every wildcard is narrowed
        away. Each source is named through both its bare and its
        current-module-anchored spelling, so retraction sees an atom already
        cleared by an earlier contribution, one a later contribution still
        shares, and one a contribution owns outright."""
        session = ReplSession()
        assert session.eval_entry(
            "scope SourceA\nrecord Point\n  x: int\ndef onlyA() -> int = 11\nend SourceA\n"
            "scope SourceB\nrecord Point\n  y: int\ndef onlyB() -> int = 22\nend SourceB\n"
        ).ok
        assert session.eval_entry(
            "scope Outer\n"
            "use SourceA::*\nuse ::SourceA::*\nuse SourceB::*\nuse ::SourceB::*\n"
            "end Outer"
        ).ok

        narrowed = session.eval_entry(
            "scope Outer\n"
            "use SourceA::{onlyA}\nuse ::SourceA::{onlyA}\n"
            "use SourceB::{onlyB}\nuse ::SourceB::{onlyB}\n"
            "end Outer"
        )
        assert narrowed.ok, narrowed.diagnostics

        result = session.eval_entry(
            "scope Outer\ndef check() -> int = onlyA() + onlyB()\nend Outer"
        )
        assert result.ok, result.diagnostics
        assert session.eval_entry("Outer::check()").value == IntValue(33)

        stale = session.eval_entry(
            "scope Outer\ndef mk() -> int = case Point(x = 1) of | Point(x) => x\nend Outer"
        )
        assert not stale.ok

    def test_local_use_of_a_scope_that_never_promotes_is_dropped_on_replay(self) -> None:
        """A retained local ``use`` whose source scope fails to promote on
        the SAME entry that declared it (first declaration, runtime failure)
        must not be replayed on a later entry -- there is no promoted source
        scope left to resolve it against."""
        session = ReplSession()
        assert session.eval_entry("scope Outer\nend Outer").ok

        failed = session.eval_entry(
            "let z: decimal = 1 / 0\n"
            "enum Color\n  | Red\n  | Blue\n"
            "scope Outer\nuse Color::*\nend Outer\n"
        )
        assert not failed.ok

        assert not session.eval_entry("Color::Red").ok


# ---------------------------------------------------------------------------
# Session bootstrap (ReplSession.open)
# ---------------------------------------------------------------------------


class TestSessionOpen:
    """``ReplSession.open`` loads the initial library image before any entry runs.

    A host (``agm repl``) calls it once, right after constructing the
    session and before printing a banner or accepting input, so a rejected
    engine-setting override is reported before the session appears to have
    started. The loaded library modules become the cache the first entry
    reuses, so opening the session never doubles the standard-library
    compile a lone first entry would otherwise perform on its own.
    """

    def test_open_with_no_overrides_preloads_the_default_stdlib(self) -> None:
        from agm.agl.modules.ids import STD_CONFIG_ID, STD_CORE_ID

        s = ReplSession()
        assert s.open() == ()
        assert STD_CORE_ID in s._loaded_lib_modules
        assert STD_CONFIG_ID in s._loaded_lib_modules
        assert s._next_node_id > 0

    def test_open_without_stdlib_is_a_harmless_no_op(self) -> None:
        s = ReplSession(default_stdlib=False)
        assert s.open() == ()
        assert s._loaded_lib_modules == {}

    def test_open_applies_a_well_formed_override_before_the_first_entry(self) -> None:
        from agm.agl.semantics.values import RecordValue, TextValue
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("preloaded")', origin="--agent"
                )
            }
        )
        assert s.open() == ()

        result = s.eval_entry("import std/config\nstd/config::default-agent")
        assert result.ok
        assert isinstance(result.value, RecordValue)
        assert result.value.display_name.rsplit("::", maxsplit=1)[-1] == "AgentCommand"
        assert result.value.fields["command"] == TextValue("preloaded")

    def test_open_rejects_an_unparseable_override_naming_its_origin(self) -> None:
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            setting_overrides={"default-agent": SettingOverride(source="(", origin="--agent")}
        )
        diagnostics = s.open()
        assert diagnostics
        from agm.agl.diagnostics import format_diagnostic

        assert any("--agent" in format_diagnostic(d) for d in diagnostics)
        # A rejected override promotes nothing, so the session is left exactly
        # as constructed rather than with a half-applied initial image.
        assert s._loaded_lib_modules == {}
        assert s._next_node_id == 0

    def test_open_rejects_a_wrong_typed_override_naming_its_origin(self) -> None:
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            setting_overrides={
                "default-agent": SettingOverride(source='"not-an-agent"', origin="--agent")
            }
        )
        diagnostics = s.open()
        assert diagnostics
        from agm.agl.diagnostics import format_diagnostic

        assert any("--agent" in format_diagnostic(d) for d in diagnostics)

    def test_open_rejects_a_non_constant_override_naming_its_origin(self) -> None:
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("not " + "constant")', origin="--agent"
                )
            }
        )
        diagnostics = s.open()
        assert diagnostics
        from agm.agl.diagnostics import format_diagnostic

        assert any("--agent" in format_diagnostic(d) for d in diagnostics)

    def test_reset_leaves_the_override_in_force(self) -> None:
        from agm.agl.semantics.values import RecordValue, TextValue
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("preloaded")', origin="--agent"
                )
            }
        )
        assert s.open() == ()
        assert s.eval_entry("import std/config").ok

        s.reset()

        result = s.eval_entry("import std/config\nstd/config::default-agent")
        assert result.ok
        assert isinstance(result.value, RecordValue)
        assert result.value.display_name.rsplit("::", maxsplit=1)[-1] == "AgentCommand"
        assert result.value.fields["command"] == TextValue("preloaded")

    def test_stdlib_is_loaded_exactly_once_across_open_and_two_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The initial image ``open`` builds is the cache every entry reuses.

        Spies on the module loader the way the ``agm exec`` guard does
        (``test_agl_pipeline_setting_overrides.py``), but counts freshly
        loaded modules per call rather than call count -- ``open`` calls
        ``build_repl_graph`` too, same as an entry, so counting calls alone
        would not distinguish "loaded the stdlib" from "reused the cache".
        """
        import agm.agl.modules.loader as loader_mod
        from agm.agl.setting_overrides import SettingOverride

        original = loader_mod.build_repl_graph
        new_module_counts: list[int] = []

        def spy(*args: object, **kwargs: object) -> object:
            result = original(*args, **kwargs)
            _graph, _next_id, new_modules = result
            new_module_counts.append(len(new_modules))
            return result

        monkeypatch.setattr(loader_mod, "build_repl_graph", spy)

        s = ReplSession(
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("preloaded")', origin="--agent"
                )
            }
        )
        assert s.open() == ()
        assert new_module_counts and new_module_counts[0] > 0

        assert s.eval_entry("1 + 1").ok
        assert s.eval_entry("2 + 2").ok

        assert new_module_counts[1:] == [0, 0]

    def test_open_rejects_a_required_override_when_std_config_never_loads(self) -> None:
        """``--no-stdlib`` never loads ``std/config``, but a ``required`` override
        (the default; mirrors a CLI ``--agent`` flag) is a request the host
        cannot silently drop -- it still fails ``open()`` up front, rather than
        being deferred to whichever entry, if any, first imports ``std/config``.
        """
        from agm.agl.diagnostics import format_diagnostic
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            default_stdlib=False,
            setting_overrides={
                "default-agent": SettingOverride(source='AgentCommand("x")', origin="--agent")
            },
        )
        diagnostics = s.open()
        assert diagnostics
        assert any("--agent" in format_diagnostic(d) for d in diagnostics)
        assert s._loaded_lib_modules == {}

    def test_open_with_non_required_override_and_no_stdlib_is_a_harmless_no_op(self) -> None:
        """A non-``required`` override (ambient configuration, e.g.
        ``[exec] default-agent``) is simply inert when ``std/config`` never
        loads: ``open()`` must not fail because of it.
        """
        from agm.agl.setting_overrides import SettingOverride

        s = ReplSession(
            default_stdlib=False,
            setting_overrides={
                "default-agent": SettingOverride(
                    source='AgentCommand("x")', origin="[exec] default-agent", required=False
                )
            },
        )
        assert s.open() == ()
        assert s._loaded_lib_modules == {}

    def test_open_reports_an_unreadable_stdlib_module_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        """A module under the stdlib root that fails to read is a diagnostic, not a crash.

        ``open`` documents "Never raises": every failure
        ``load_and_check_program`` can produce must come back as a diagnostic
        tuple, the same as it already does for a syntax/scope/type error. A
        module file's own I/O failure -- invalid UTF-8 here, the same class of
        failure a permission-denied file would raise via
        ``agm.core.fs.read_text`` -- was previously left uncaught, an
        unhandled ``UnicodeDecodeError`` breaking the "Never raises" contract.
        """
        std_dir = tmp_path / "std"
        std_dir.mkdir()
        (std_dir / "core.agl").write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")
        s = ReplSession(stdlib_root=tmp_path)

        diagnostics = s.open()

        assert diagnostics
        assert s._loaded_lib_modules == {}


class TestDeferredStdlibResolution:
    """Constructing a session without an explicit ``stdlib_root`` stays total."""

    def test_construction_does_not_raise_against_a_version_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.config.module_roots as module_roots
        from agm.config.module_roots import StdlibVersionMismatchError

        def _raise_version_mismatch(*, home: Path) -> Path:
            raise StdlibVersionMismatchError("0.0.1", "0.1.0")

        monkeypatch.delenv("AGM_STDLIB", raising=False)
        monkeypatch.setattr(module_roots, "resolve_stdlib_root", _raise_version_mismatch)

        # Must not raise: resolution has not happened yet.
        s = ReplSession()

        # The lazy failure is caught by ``open``'s own "Never raises" contract.
        diagnostics = s.open()
        assert diagnostics
        assert "0.0.1" in diagnostics[0].message

    def test_default_stdlib_resolution_honors_agm_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A relocated ``AGM_HOME`` -- not the process home directory -- selects the stdlib.

        Mirrors how ``commands/repl.py`` resolves this session's stdlib root
        (``current_config_context(...).home``): the deferred resolution in
        ``_ensure_roots`` must use the same seam, not a hardcoded
        ``Path.home()``, so a relocated ``AGM_HOME`` is honored here exactly
        as it is for every other AGM home lookup.
        """
        import semver

        from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
        from agm.packages.record import write_record
        from agm.version import AGM_VERSION

        agm_home = tmp_path / "relocated-agm"
        stdlib_root = agm_home / "packages" / "std" / AGM_VERSION
        std_dir = stdlib_root / "std"
        std_dir.mkdir(parents=True)
        real_stdlib = Path(__file__).resolve().parents[1] / "stdlib" / "std"
        for name in ("core.agl", "config.agl", "fs.agl", "fs.py", "text.agl", "text.py"):
            source = real_stdlib / name
            (std_dir / name).write_bytes(source.read_bytes())
        (stdlib_root / "package.toml").write_bytes(
            (Path(__file__).resolve().parents[1] / "stdlib" / "package.toml").read_bytes()
        )
        write_record(stdlib_root)
        write_activation_index(
            ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
            home=tmp_path / "unused-home",
            env={"AGM_HOME": str(agm_home)},
        )

        monkeypatch.delenv("AGM_STDLIB", raising=False)
        monkeypatch.setenv("AGM_HOME", str(agm_home))

        s = ReplSession()
        assert s.open() == ()

        assert s._roots is not None
        assert stdlib_root.resolve() in s._roots.roots
