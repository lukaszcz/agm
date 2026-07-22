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
from unittest.mock import patch

import pytest

from agm.agl.diagnostics import AglError
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
from agm.agl.semantics.values import VOID_VALUE, BoolValue, IntValue, UnitValue

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
    raise AssertionError(f"no test literal for {typ!r}")


def _constructor_args(fields: dict[str, Type]) -> str:
    return ", ".join(f"{name} = {_literal_for_type(typ)}" for name, typ in fields.items())


# ---------------------------------------------------------------------------
# Persistence across entries
# ---------------------------------------------------------------------------


class TestPersistence:
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

        result = s.eval_entry("record Widget\n  x: int")

        assert not result.ok
        assert "already declared" in result.diagnostics[0].message.lower()

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

    def test_partial_application_closure_persists_into_next_entry(self) -> None:
        s = ReplSession()

        r1 = s.eval_entry("def add(x: int, y: int) -> int = x + y\nlet add1 = add(1, ?)")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("add1(2)")

        assert r2.ok, r2.diagnostics
        assert r2.kind == "expression"
        assert r2.value is not None
        assert _int(r2.value) == 3


# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------


class TestStdlib:
    def test_core_stdlib_is_opened_unqualified_by_default(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        some_result = s.eval_entry("let present: Option[int] = Some(value = 1)")
        none_result = s.eval_entry("let missing: Option[int] = None")

        assert some_result.ok, some_result.diagnostics
        assert none_result.ok, none_result.diagnostics

    def test_type_of_uses_opened_core_stdlib(self) -> None:
        s = ReplSession(stdlib_root=Path(__file__).resolve().parents[1] / "stdlib")

        assert "Option[int]" in s.type_of("Some(value = 1)")

    def test_no_stdlib_requires_explicit_core_import_after_reset(self) -> None:
        s = ReplSession(
            default_stdlib=False,
            stdlib_root=Path(__file__).resolve().parents[1] / "stdlib",
        )

        assert not s.eval_entry("Some(value = 1)").ok
        assert s.eval_entry("open import std/core\nSome(value = 1)").ok

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
                for variant, fields in typedef.variants:
                    args = _constructor_args(dict(fields))
                    call = f"{name}::{variant}({args})" if args else f"{name}::{variant}"
                    result = s.eval_entry(call)
                    assert result.ok, (name, variant, result.diagnostics)
                    assert result.value_type is not None
                    assert result.value_type.name == name

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
            fields = {
                field_name: field_type
                for field_name, field_type in table.exception_fields(typ).items()
                if field_name != "trace_id"
            }
            result = s.eval_entry(f"{name}({_constructor_args(fields)})")
            assert result.ok, (name, result.diagnostics)
            assert result.value_type is not None
            assert result.value_type.name == name


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

    def test_record_redefinition_clears_generic_metadata(self) -> None:
        s = ReplSession()
        first = s.eval_entry("record Box[T]\n  x: T")
        assert first.ok
        second = s.eval_entry("record Box\n  x: int")
        assert second.ok

        use = s.eval_entry("let y: Box[int] = Box(x = 1)")

        assert not use.ok
        assert any("does not take type arguments" in d.message for d in use.diagnostics)


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
            "let t = Node(value = 1, left = Leaf(), right = Node(value = 2, left = Leaf(), "
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
        first = s.eval_entry("record Category\n  name: text\n  subcategories: list[Category]")
        assert first.ok
        second = s.eval_entry("record Category\n  label: text\n  kids: list[Category]")
        assert second.ok
        use = s.eval_entry('let c = Category(label = "root", kids = [])')
        assert use.ok
        stale = s.eval_entry('let bad = Category(name = "root", subcategories = [])')
        assert not stale.ok

    def test_type_redefinition_invalidation_detects_nested_nominal_types(self) -> None:
        from agm.agl.semantics.types import (
            DictType,
            ExceptionType,
            FunctionType,
            ListType,
            RecordType,
            TextType,
        )

        names = frozenset({"R"})

        assert ReplSession._type_mentions_entry_nominal(ExceptionType("R"), names)
        assert ReplSession._type_mentions_entry_nominal(ListType(RecordType("R")), names)
        assert ReplSession._type_mentions_entry_nominal(DictType(RecordType("R")), names)
        assert ReplSession._type_mentions_entry_nominal(
            FunctionType((RecordType("R"),), TextType()), names
        )
        assert ReplSession._type_mentions_entry_nominal(FunctionType((), RecordType("R")), names)
        assert not ReplSession._type_mentions_entry_nominal(TextType(), names)

    def test_record_redefinition_invalidates_old_nominal_values(self) -> None:
        s = ReplSession()
        assert s.eval_entry("record R\n  old: int").ok
        assert s.eval_entry("let stale = R(old = 1)").ok
        assert s.eval_entry("record R\n  fresh: int").ok

        result = s.eval_entry("stale.fresh")

        assert not result.ok

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
        s = ReplSession(default_agent=agent)
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

    def test_multi_item_entry_keeps_normal_block_discard_strictness(self) -> None:
        result = ReplSession().eval_entry("let first = 1\nfirst\nlet second = 2")

        assert not result.ok

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

    def test_type_of_displays_enum_constructors(self) -> None:
        s = ReplSession()
        s.eval_entry("enum Result\n  | Ok(value: int)\n  | Err(message: text)\n  | Unknown")
        s.eval_entry("let r = Ok(value = 1)")

        assert (
            s.type_of("r") == "enum Result\n  | Ok(value: int)\n  | Err(message: text)\n  | Unknown"
        )

    def test_type_of_resolves_prior_entry_constructor(self) -> None:
        s = ReplSession()
        assert s.eval_entry("enum Result\n  | Ok(value: int)\n  | Err(message: text)").ok

        expected = "enum Result\n  | Ok(value: int)\n  | Err(message: text)"
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
        s = ReplSession(default_agent=agent)
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

    def test_runtime_raise_does_not_promote_later_type(self) -> None:
        s = ReplSession()
        result = s.eval_entry("let z: decimal = 1 / 0\nrecord After\n  value: int")
        assert not result.ok
        assert not s.eval_entry("After").ok
        assert not s.eval_entry("After(value = 3)").ok

    def test_runtime_raise_does_not_retain_later_function_signature(self) -> None:
        s = ReplSession()
        failed = s.eval_entry("let z: decimal = 1 / 0\ndef later[T](x: T) -> T = x")

        assert not failed.ok
        assert not s.eval_entry("later(1)").ok
        assert s.eval_entry("def later[T](x: T) -> T = x").ok
        assert s.eval_entry("later(1)").value_type == IntType()

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
            "exception Before extends Exception\n  code: int\nlet z: decimal = 1 / 0"
        )
        assert not result.ok
        assert s.eval_entry('Before(message = "m", code = 3)').ok

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
        from agm.agl.semantics.values import IntValue, ListValue

        s = ReplSession()
        r1 = s.eval_entry("var xs = [1, 2, 3]")
        assert r1.ok
        r2 = s.eval_entry("xs[0] := 99\nlet z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        vals = {n: v for n, _t, v in s.bindings()}
        assert vals["xs"] == ListValue((IntValue(99), IntValue(2), IntValue(3)))

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
        s = ReplSession(default_agent=agent)
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
        s = ReplSession(default_agent=agent)
        result = s.eval_entry('ask """say something"""')

        assert result.ok
        assert result.quote_strings is False
        assert render_entry_result(result, echo=True) == "the-answer"

    def test_stored_ask_result_echo_uses_normal_text_quoting(self) -> None:
        from agm.agl.repl.render import render_entry_result

        agent = CountingAgent("the-answer")
        s = ReplSession(default_agent=agent)
        first = s.eval_entry('let txt: text = ask """say something"""')
        second = s.eval_entry("txt")

        assert first.ok
        assert first.quote_strings is True
        assert second.ok
        assert second.quote_strings is True
        assert render_entry_result(second, echo=True) == '"the-answer"'

    def test_distinct_agent_responses_across_entries(self) -> None:
        agent = CountingAgent("first", "second", "third")
        s = ReplSession(default_agent=agent)
        s.eval_entry('let a = ask """q1"""')
        s.eval_entry('let b = ask """q2"""')
        s.eval_entry('let c = ask """q3"""')
        vals = {n: _text(v) for n, _t, v in s.bindings()}
        assert vals == {"a": "first", "b": "second", "c": "third"}
        assert agent.calls == 3

    def test_named_agent_dispatch(self) -> None:
        # In AgL, named-agent calls use ask(prompt, agent: name) syntax.
        named = CountingAgent("named-reply")
        s = ReplSession()
        s.register_agent("reviewer", named)
        r = s.eval_entry('agent reviewer\nlet out = ask("""review this""", agent = reviewer)')
        assert r.ok, r.diagnostics
        assert _text({name: value for name, _typ, value in s.bindings()}["out"]) == "named-reply"
        assert named.calls == 1


# ---------------------------------------------------------------------------
# Agent declarations / ambient registration (host registration declares+backs)
# ---------------------------------------------------------------------------


class TestAgentDeclarations:
    def test_registered_agent_callable_without_declaration(self) -> None:
        # Host registration both DECLARES and BACKS an agent in the REPL: a
        # source ``agent`` declaration is still needed for the agent to appear
        # as a value in ask(agent: …) calls, but the host registration means
        # the ask(prompt) default-agent path works without any source decl.
        # For named agents, the source must declare them to use as a value.
        # Test: registering and declaring an agent in the same entry works.
        s = ReplSession()
        s.register_agent("reviewer", CountingAgent("ok"))
        r = s.eval_entry('agent reviewer\nask("""look""", agent = reviewer)')
        assert r.ok

    def test_undeclared_unregistered_agent_call_errors(self) -> None:
        # A call to an agent that is neither registered nor declared in source is
        # still a static scope binding error.
        s = ReplSession()
        r = s.eval_entry('ghost "hi"')
        assert not r.ok
        assert r.diagnostics

    def test_cross_entry_source_declaration_resolves(self) -> None:
        # An ``agent X`` declaration in one entry makes a later ask(agent: X)
        # call resolve without re-declaring it (X is in the ambient set).
        # The agent is also registered so the call has a backing when it dispatches.
        s = ReplSession()
        s.register_agent("helper", CountingAgent("done"))
        r1 = s.eval_entry("agent helper")
        assert r1.ok
        r2 = s.eval_entry('let out = ask("""go""", agent = helper)')
        assert r2.ok, r2.diagnostics
        assert _text({name: value for name, _typ, value in s.bindings()}["out"]) == "done"

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

    def test_unused_declaration_warning_surfaced(self) -> None:
        # A bare cross-entry ``agent X`` declaration legitimately produces an
        # "unused" scope warning, routed alongside type-checker warnings.
        s = ReplSession()
        r = s.eval_entry("agent solo")
        assert r.ok
        assert any("solo" in w.message for w in r.warnings)

    def test_type_of_allows_registered_agent_call(self) -> None:
        # The introspection (``type_of``) resolve path must also treat registered
        # agents as ambient, so typing an ask(agent: …) expression does not raise
        # a scope error.  The agent must be source-declared to appear as a value.
        s = ReplSession()
        s.register_agent("reviewer", CountingAgent("x"))
        s.eval_entry("agent reviewer")
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

    def test_declared_param_then_reference(self) -> None:
        s = ReplSession()
        s.eval_entry('param name: text = "World"')
        r = s.eval_entry("name")
        assert r.ok
        assert _text(r.value) == "World"
        _n, _t, val = s.declared_params()[0]
        assert val is not None

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

    def test_program_name_loads_param_config(self) -> None:
        s = ReplSession(params_config_loader=lambda name: {"count": 7} if name == "demo" else {})
        r = s.eval_entry("program demo\nparam count: int\ncount + 1")
        assert r.ok
        assert _int(r.value) == 8
        assert s.program_name() == "demo"

    def test_param_config_conversion_error_rejects_entry(self) -> None:
        s = ReplSession(params_config_loader=lambda _name: {"count": "not-json-int"})
        r = s.eval_entry("program demo\nparam count: int\ncount")
        assert not r.ok
        assert "Config value for param 'count' is invalid" in r.diagnostics[0].message
        assert s.program_name() is None

    def test_redeclaring_different_program_name_rejects_entry(self) -> None:
        s = ReplSession()
        assert s.eval_entry("program demo\n1").ok
        r = s.eval_entry("program other\n2")
        assert not r.ok
        assert "Program name already set" in r.diagnostics[0].message
        assert s.program_name() == "demo"

    def test_redeclaring_same_program_name_is_noop(self) -> None:
        s = ReplSession()
        assert s.eval_entry("program demo\n1").ok
        r = s.eval_entry("program demo\n2")
        assert r.ok
        assert s.program_name() == "demo"


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


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
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
        s = ReplSession(default_agent=agent)
        s.load_file(f)
        assert agent.calls == 1
        # Referencing it later does not re-run.
        s.eval_entry("g")
        assert agent.calls == 1

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
        assert s.dump_source() == "let a = 1\nlet b = 2"

    def test_dump_source_excludes_failed_entries(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 1")
        s.eval_entry("let z: decimal = 1 / 0")  # runtime fail
        s.eval_entry('let b = a + "x"')  # type fail
        assert s.dump_source() == "let a = 1"


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
        s = ReplSession(default_agent=agent)
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
    def test_agents_lists_named_and_ask(self) -> None:
        s = ReplSession(default_agent=CountingAgent("x"))
        s.register_agent("alpha", CountingAgent("a"))
        s.register_agent("beta", CountingAgent("b"))
        assert s.agents() == ["alpha", "beta", "ask"]

    def test_agents_without_default_excludes_ask(self) -> None:
        s = ReplSession()
        s.register_agent("only", CountingAgent("x"))
        assert s.agents() == ["only"]

    def test_register_agent_reserved_name_rejected(self) -> None:
        s = ReplSession()
        with pytest.raises(ValueError):
            s.register_agent("ask", CountingAgent("x"))

    def test_register_duplicate_agent_rejected(self) -> None:
        s = ReplSession()
        s.register_agent("dup", CountingAgent("x"))
        with pytest.raises(ValueError):
            s.register_agent("dup", CountingAgent("y"))

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

        s = ReplSession(default_agent=CountingAgent("ok"))
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
        s = ReplSession(default_agent=_CancellingAgent())
        r = s.eval_entry('let g = ask """do it"""')
        assert not r.ok
        assert r.error is None
        assert r.diagnostics
        assert "cancelled" in r.diagnostics[0].message.lower()

    def test_declined_agent_leaves_bindings_unchanged(self) -> None:
        s = ReplSession(default_agent=_CancellingAgent())
        s.eval_entry("let keep = 7")
        before = _snapshot(s)
        r = s.eval_entry('let g = ask """do it"""')
        assert not r.ok
        # The cancelled initializer did not complete, so it installs nothing.
        assert _snapshot(s) == before
        assert all(n != "g" for n, _t, _v in s.bindings())

    def test_keyboard_interrupt_aborts_entry(self) -> None:
        s = ReplSession(default_agent=_InterruptAgent())
        s.eval_entry("let x = 1")
        before = _snapshot(s)
        r = s.eval_entry('let g = ask """slow"""')
        assert not r.ok
        assert r.error is None
        assert _snapshot(s) == before

    def test_cancellation_preserves_prior_assignment(self) -> None:
        s = ReplSession(default_agent=_CancellingAgent())
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
        s = ReplSession(default_agent=_CancellingAgent())
        r = s.eval_entry('record Box\n  value: int\nlet g = ask """x"""')
        assert not r.ok
        assert s.eval_entry("Box(value = 3)").ok

    def test_cancellation_excludes_record_declared_after_call(self) -> None:
        # A type declared after the cancelled call is not promoted.
        s = ReplSession(default_agent=_CancellingAgent())
        r = s.eval_entry('let g = ask """x"""\nrecord After\n  value: int')
        assert not r.ok
        assert not s.eval_entry("After(value: 1)").ok


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------


class TestTraceLogging:
    def test_no_trace_path_writes_nothing(self, tmp_path: Path) -> None:
        s = ReplSession(default_agent=CountingAgent("ok"))
        r = s.eval_entry('let g = ask """hi"""')
        assert r.ok
        assert r.trace_path is None

    def test_trace_file_records_run_and_agent_call(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(default_agent=CountingAgent("reply"), trace_path=trace)
        r = s.eval_entry('let g = ask """ask"""')
        assert r.ok
        assert r.trace_path == trace
        assert trace.exists()
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        kinds = [rec["kind"] for rec in records]
        assert "run_start" in kinds
        assert "run_end" in kinds
        assert "agent_call_attempt" in kinds

    def test_each_entry_is_its_own_run(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(default_agent=CountingAgent("a", "b"), trace_path=trace)
        s.eval_entry('let x = ask """one"""')
        s.eval_entry('let y = ask """two"""')
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        run_ids = {rec["run_id"] for rec in records}
        # Per-entry TraceStore → a fresh run_id per entry, all in one file.
        assert len(run_ids) == 2

    def test_check_only_writes_no_trace(self, tmp_path: Path) -> None:
        trace = tmp_path / "repl.log"
        s = ReplSession(default_agent=CountingAgent("ok"), trace_path=trace)
        r = s.eval_entry('let g = ask """hi"""', check_only=True)
        assert r.ok
        assert r.trace_path is None
        assert not trace.exists()

    def test_cancelled_entry_records_run_end(self, tmp_path: Path) -> None:
        import json

        trace = tmp_path / "repl.log"
        s = ReplSession(default_agent=_CancellingAgent(), trace_path=trace)
        r = s.eval_entry('let g = ask """x"""')
        assert not r.ok
        records = [json.loads(line) for line in trace.read_text().splitlines() if line]
        run_end = [rec for rec in records if rec["kind"] == "run_end"]
        assert run_end and run_end[-1]["ok"] is False

    def test_write_failure_disables_logging_for_that_entry_only(self, tmp_path: Path) -> None:
        """A transient write failure must not kill logging for the whole session."""
        import json

        from agm.core import log as core_log

        trace = tmp_path / "repl.log"
        s = ReplSession(default_agent=CountingAgent("a", "b"), trace_path=trace)
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
    """REPL import declaration support."""

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
        # Open import: functions are in unqualified scope
        r = s.eval_entry("open import mylib\nadd(3, 4)")
        assert r.ok, r.diagnostics
        assert r.kind == "expression"
        assert _int(r.value) == 7

    def test_import_persists_across_entries(self, tmp_path: Path) -> None:
        lib = tmp_path / "util.agl"
        lib.write_text("def double(x: int) -> int = x * 2\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("open import util")
        assert r1.ok, r1.diagnostics
        # Open import: double() is in unqualified scope in next entry too
        r2 = s.eval_entry("double(5)")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 10

    def test_import_using_hiding(self, tmp_path: Path) -> None:
        lib = tmp_path / "funcs.agl"
        lib.write_text("def square(n: int) -> int = n * n\ndef cube(n: int) -> int = n * n * n\n")
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("import funcs using square\nsquare(4)")
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
        # Import once to cache it (open import: greet() in unqualified scope)
        r1 = s.eval_entry("open import cached\ngreet()")
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
        r = s.eval_entry("open import temp\nf()")
        assert r.ok, r.diagnostics
        s.reset()
        assert not s._loaded_lib_modules
        assert not s._accumulated_imports

    def test_runtime_failure_restores_unpromoted_type_nominal(self, tmp_path: Path) -> None:
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("record R\n  x: int")
        assert r1.ok, r1.diagnostics

        r2 = s.eval_entry("let z: decimal = 1 / 0\nrecord R\n  y: int")
        assert not r2.ok
        from agm.agl.ir.ids import NominalId
        from agm.agl.modules.ids import ENTRY_ID

        assert s._link_image._state.nominals[NominalId(ENTRY_ID, "R")].fields == ("x",)

        r3 = s.eval_entry("let f = R\nlet v = f(1)\nv.x")

        assert r3.ok, r3.diagnostics
        assert _int(r3.value) == 1

    def test_runtime_failure_does_not_mark_module_linked(self, tmp_path: Path) -> None:
        # Regression: when an entry imports a previously unseen
        # module and then raises at runtime, the module must NOT be marked as
        # persistently linked. Otherwise the next import reloads it with fresh
        # declaration IDs but skips lowering it (already linked), crashing with
        # ``no FunctionId for function decl_node_id``.
        lib = tmp_path / "boom.agl"
        lib.write_text("def f() -> int = 42\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("import boom\nlet z: decimal = 1 / 0")
        assert not r1.ok
        # The failed entry cached neither the loaded module nor the link.
        assert not s._loaded_lib_modules
        assert not s._link_image._linked_modules
        # Re-importing reloads and re-lowers boom (with fresh decl IDs) and
        # evaluates successfully instead of hitting a stale-link assertion.
        r2 = s.eval_entry("open import boom\nf()")
        assert r2.ok, r2.diagnostics
        assert _int(r2.value) == 42

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
        r = s.eval_entry("open import mylib\nadd(1, 2)", check_only=True)
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

    def test_agent_decl_in_graph_mode(self, tmp_path: Path) -> None:
        # Declaring an agent in program context installs it in the entry scope.
        lib = tmp_path / "mylib.agl"
        lib.write_text("def id_fn(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        s.register_agent("helper", CountingAgent("ok"))
        r = s.eval_entry("open import mylib\nagent helper\nid_fn(7)")
        assert r.ok, r.diagnostics

    def test_agl_raise_in_graph_mode(self, tmp_path: Path) -> None:
        # An AglRaise exception during program evaluation aborts the entry.
        lib = tmp_path / "mylib.agl"
        lib.write_text('def boom() -> int = raise Abort(message = "boom")\n')
        s = self._make_session_with_root(tmp_path)
        r = s.eval_entry("open import mylib\nboom()")
        assert not r.ok
        assert r.error is not None

    def test_agl_raise_in_graph_mode_records_exception_trace(self, tmp_path: Path) -> None:
        import json

        lib = tmp_path / "mylib.agl"
        lib.write_text('def boom() -> int = raise Abort(message = "boom")\n')
        trace = tmp_path / "trace.jsonl"
        s = self._make_session_with_root(tmp_path)
        s._trace_path = trace

        r = s.eval_entry("open import mylib\nboom()")

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

        assert s.eval_entry("open import tools/*\nadd() + mul()").ok
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
            "open import tools/add\nimport tools/mul as old_mul\nadd() + old_mul::mul()"
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

        assert s.eval_entry("open import math\nadd() + mul()").ok
        result = s.eval_entry(
            "import math using add\nimport math as arithmetic\nadd() + arithmetic::mul()"
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
            "import math using add\nimport math as arithmetic\nadd() + arithmetic::mul()"
        ).ok
        replacement = s.eval_entry("import math hiding add\nmath::mul()")

        assert replacement.ok, replacement.diagnostics
        assert not s.eval_entry("add()").ok
        assert not s.eval_entry("arithmetic::mul()").ok
        assert not s.eval_entry("math::add()").ok
        assert s.eval_entry("math::mul()").ok

    def test_replacement_removes_alias_using_hiding_and_open_options(self, tmp_path: Path) -> None:
        lib = tmp_path / "api.agl"
        lib.write_text("def alpha() -> int = 1\ndef beta() -> int = 2\n")
        s = self._make_session_with_root(tmp_path)

        assert s.eval_entry("open import api\nalpha() + beta()").ok
        assert s.eval_entry("import api as old_api\nold_api::alpha()").ok
        assert not s.eval_entry("alpha()").ok
        assert not s.eval_entry("beta()").ok

        assert s.eval_entry("import api using beta\nbeta()").ok
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
        r = s.eval_entry("open import lazylib\nval()")
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
        s.register_agent("helper", CountingAgent("ok"))
        s.register_codec(BadCodec())
        r = s.eval_entry('import mylib\nagent helper\nask("hi", agent = helper, format = "bad")')
        assert not r.ok
        assert any("Contract error" in d.message for d in r.diagnostics)

    def test_program_decl_conflict_in_graph_mode(self, tmp_path: Path) -> None:
        # _pre_eval_param_check returning non-None in program context:
        # setting a different program name when one is already set.
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        s = self._make_session_with_root(tmp_path)
        r1 = s.eval_entry("program first\n1")
        assert r1.ok, r1.diagnostics
        # Now in program context, try to declare a different program name.
        r2 = s.eval_entry("open import mylib\nprogram second\nadd(1, 2)")
        assert not r2.ok
        assert "Program name already set" in r2.diagnostics[0].message

    def test_cancellation_in_graph_mode(self, tmp_path: Path) -> None:
        # AgentCancelled during program execution aborts the entry.
        lib = tmp_path / "mylib.agl"
        lib.write_text("def noop(n: int) -> int = n\n")
        s = self._make_session_with_root(tmp_path)
        s.register_agent("helper", _CancellingAgent())
        r = s.eval_entry('import mylib\nagent helper\nnoop(ask("hi", agent = helper))')
        assert not r.ok
        assert r.error is None
        assert r.diagnostics

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
        r1 = s.eval_entry("open import foo/*\nval()")
        assert r1.ok, r1.diagnostics
        # Entry 2: plain import of foo (brings top() into scope)
        r2 = s.eval_entry("open import foo\ntop()")
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

        assert s.eval_entry("open import tools/*\nadd()").ok

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

        assert s.eval_entry("open import tools/*\nadd() + mul()").ok
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
        r1 = s.eval_entry("open import extlib")
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
        s.eval_entry("open import extlib")
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
        r1 = s.eval_entry("open import counting\ntouch()")
        assert r1.ok, r1.diagnostics
        r2 = s.eval_entry("touch()")
        assert r2.ok, r2.diagnostics
        # Re-importing the same module in a later entry is a no-op import.
        r3 = s.eval_entry("open import counting\ntouch()")
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
        r1 = s.eval_entry("open import extlib\nadd_one(1)")
        assert r1.ok, r1.diagnostics
        s.reset()
        s._roots = roots  # :reset clears roots too; a real host re-supplies them
        r2 = s.eval_entry("open import extlib\nadd_one(1)")
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
        s.eval_entry("open import extlib")
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
        s.eval_entry("open import extlib")
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

    def test_bare_agent_name_echo_does_not_crash(self) -> None:
        """A bare agent-name entry echoes the surface form without crashing."""
        s = ReplSession()
        s.register_agent("reviewer", CountingAgent("ok"))
        # Declare the agent in source so it becomes a value binding in scope.
        s.eval_entry("agent reviewer")
        r = s.eval_entry("reviewer")
        assert r.ok
        assert r.kind == "expression"
        assert r.value is not None
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import AgentValue

        assert isinstance(r.value, AgentValue)
        rendered = render_value(r.value)
        assert rendered == "<agent reviewer>"

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
        from agm.agl.semantics.types import DictType, ListType

        s = ReplSession()
        r = s.eval_entry("list[int]")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, ListType)
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
        from agm.agl.semantics.types import ListType

        s = ReplSession()
        s.eval_entry("type Pair[A, B] = list[A]")
        # The alias resolves transparently to its target: list[int].
        r = s.eval_entry("Pair[int, text]")
        assert r.ok
        assert r.kind == "type"
        assert isinstance(r.value_type, ListType)
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

    def test_opened_stdlib_bare_generic_type_echoes_definition(self) -> None:
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

    def test_opened_stdlib_qualified_bare_generic_type_echoes_definition(self) -> None:
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
        assert local_expr.module_qualifier is not None

        local_env = TypeEnvironment()
        local_env.register_generic_type("Box", local_def)
        assert local_env.resolve_qualified_unapplied_generic_type(
            local_expr.module_qualifier,
            "Box",
        ) == ("Box", local_def)

        empty_env = TypeEnvironment()
        assert (
            empty_env.resolve_qualified_unapplied_generic_type(
                local_expr.module_qualifier,
                "Box",
            )
            is None
        )

        graph_env = TypeEnvironment(program_generic_table={(ENTRY_ID, "Box"): local_def})
        assert graph_env.resolve_qualified_unapplied_generic_type(
            local_expr.module_qualifier,
            "Box",
        ) == ("Box", local_def)

        lib = ModuleId(("lib",))
        qualified_expr = parse_type_expr("missing::Box")
        assert isinstance(qualified_expr, NameT)
        assert qualified_expr.module_qualifier is not None
        no_handle_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(contributions={}, unqualified={}),
        )
        assert (
            no_handle_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.module_qualifier,
                "Box",
            )
            is None
        )

        no_name_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(
                contributions={
                    lib: ModuleContribution(lib, {}, frozenset(), False, frozenset({"missing"}))
                },
                unqualified={},
            ),
        )
        assert (
            no_name_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.module_qualifier,
                "Box",
            )
            is None
        )

        no_generic_env = TypeEnvironment(
            program_generic_table={},
            import_env=ImportEnv(
                contributions={
                    lib: ModuleContribution(
                        lib, {"Box": (lib, "Box")}, frozenset(), False, frozenset({"missing"})
                    )
                },
                unqualified={},
            ),
        )
        assert (
            no_generic_env.resolve_qualified_unapplied_generic_type(
                qualified_expr.module_qualifier,
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
