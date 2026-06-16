"""Tests for the UI-free REPL session core (``agm.agl.repl.session``).

Drives ``ReplSession`` directly with source strings and fake agents.  Asserts
user-visible behaviour: persistence across entries, redefinition/shadowing,
expression/binding echo data, ``type_of`` purity, atomic-on-error promotion,
exactly-once agent dispatch, the ``:set`` input flow, ``reset``, ``load_file``,
``dump_source``, surfaced warnings, and ``check_only`` (type-only) runs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from agm.agl.diagnostics import AglError
from agm.agl.repl import EntryResult, ReplSession
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.typecheck.types import IntType, TextType

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


# ---------------------------------------------------------------------------
# Persistence across entries
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_binding_persists_into_next_entry(self) -> None:
        s = ReplSession()
        r1 = s.eval_entry("let x = 1 + 2")
        assert r1.ok
        r2 = s.eval_entry("let y = x * 10")
        assert r2.ok
        names = {n: v for n, _t, v in s.bindings()}
        assert {"x", "y"} <= set(names)

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
        use = s.eval_entry('let r = R(b: "hi")')
        assert use.ok
        bad = s.eval_entry("let r2 = R(a: 1)")
        assert not bad.ok  # old field 'a' no longer valid


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

    def test_binding_echo_data(self) -> None:
        s = ReplSession()
        r = s.eval_entry("let total = 5")
        assert r.kind == "binding"
        assert r.name == "total"
        assert r.value is not None and _int(r.value) == 5
        assert isinstance(r.value_type, IntType)

    def test_declaration_echo_kind(self) -> None:
        s = ReplSession()
        r = s.eval_entry("type Age = int")
        assert r.kind == "declaration"
        assert r.name == "Age"
        assert r.value is None

    def test_statement_echo_kind(self) -> None:
        s = ReplSession()
        r = s.eval_entry("print 1")
        assert r.kind == "statement"
        assert r.value is None
        assert r.ok


# ---------------------------------------------------------------------------
# type_of — type without evaluation, no state change, no agent
# ---------------------------------------------------------------------------


class TestTypeOf:
    def test_type_of_returns_canonical_type(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        assert s.type_of("x + 1") == repr(IntType())

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


# ---------------------------------------------------------------------------
# Atomic-on-error
# ---------------------------------------------------------------------------


class TestAtomicOnError:
    def test_type_error_leaves_bindings_unchanged(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 10")
        before = _snapshot(s)
        r = s.eval_entry('let b = a + "oops"')
        assert not r.ok
        assert r.diagnostics
        assert r.error is None
        assert _snapshot(s) == before

    def test_runtime_raise_leaves_bindings_unchanged(self) -> None:
        s = ReplSession()
        s.eval_entry("let a = 10")
        before = _snapshot(s)
        r = s.eval_entry("let z: decimal = 1 / 0")
        assert not r.ok
        assert r.error is not None  # mapped RunError, not a pre-exec diagnostic
        assert r.diagnostics == []
        assert _snapshot(s) == before

    def test_runtime_raise_rolls_back_set_to_prior_var(self) -> None:
        # A ``set`` of a prior session ``var`` mutates the persistent value scope
        # in place; a later raise in the SAME entry must roll that mutation back.
        s = ReplSession()
        r1 = s.eval_entry("var v = 1")
        assert r1.ok
        r2 = s.eval_entry("set v = 99\nlet z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["v"] == 1  # rolled back, NOT 99

    def test_successful_set_to_prior_var_persists(self) -> None:
        # The positive counterpart: a successful ``set`` in a later entry DOES
        # persist into the session.
        s = ReplSession()
        s.eval_entry("var v = 1")
        r = s.eval_entry("set v = 5")
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
        named = CountingAgent("named-reply")
        s = ReplSession()
        s.register_agent("reviewer", named)
        r = s.eval_entry('let out = reviewer """review this"""')
        assert r.ok
        assert _text(r.value) == "named-reply"
        assert named.calls == 1


# ---------------------------------------------------------------------------
# Agent declarations / ambient registration (host registration declares+backs)
# ---------------------------------------------------------------------------


class TestAgentDeclarations:
    def test_registered_agent_callable_without_declaration(self) -> None:
        # Host registration both DECLARES and BACKS an agent in the REPL: a call
        # needs no in-source ``agent`` declaration and raises no scope error.
        s = ReplSession()
        s.register_agent("reviewer", CountingAgent("ok"))
        r = s.eval_entry('reviewer "look"')
        assert r.ok

    def test_undeclared_unregistered_agent_call_errors(self) -> None:
        # A call to an agent that is neither registered nor declared in source is
        # still a static scope binding error.
        s = ReplSession()
        r = s.eval_entry('ghost "hi"')
        assert not r.ok
        assert r.diagnostics

    def test_cross_entry_source_declaration_resolves(self) -> None:
        # An ``agent X`` declaration in one entry makes a later call to X resolve
        # without re-declaring it.  The agent is also registered so the call has
        # a backing when it actually dispatches.
        s = ReplSession()
        s.register_agent("helper", CountingAgent("done"))
        r1 = s.eval_entry("agent helper")
        assert r1.ok
        r2 = s.eval_entry('let out = helper "go"')
        assert r2.ok
        assert _text(r2.value) == "done"

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
        # agents as ambient, so typing an agent-calling expression does not raise
        # a scope error.
        s = ReplSession()
        s.register_agent("reviewer", CountingAgent("x"))
        assert s.type_of('reviewer "ask"') == repr(TextType())

    def test_reset_clears_declared_agents(self) -> None:
        # After reset, a previously source-declared agent is gone: a call to it
        # (without re-registration/re-declaration) is a scope error again.
        s = ReplSession()
        s.eval_entry("agent transient")
        s.reset()
        r = s.eval_entry('transient "hi"')
        assert not r.ok


# ---------------------------------------------------------------------------
# Inputs / :set flow
# ---------------------------------------------------------------------------


class TestInputs:
    def test_declared_input_listed_unset(self) -> None:
        s = ReplSession()
        s.eval_entry("input name: text")
        ins = s.inputs()
        assert len(ins) == 1
        name, typ, val = ins[0]
        assert name == "name"
        assert isinstance(typ, TextType)
        assert val is None

    def test_unset_input_reference_is_clean_error(self) -> None:
        s = ReplSession()
        s.eval_entry("input name: text")
        r = s.eval_entry("name")
        assert not r.ok
        assert r.diagnostics
        assert "name" in r.diagnostics[0].message
        assert ":set" in r.diagnostics[0].message

    def test_set_input_then_reference(self) -> None:
        s = ReplSession()
        s.eval_entry("input name: text")
        s.set_input("name", "World")
        r = s.eval_entry("name")
        assert r.ok
        assert _text(r.value) == "World"
        # inputs() now reports the value.
        _n, _t, val = s.inputs()[0]
        assert val is not None

    def test_set_input_typed_conversion(self) -> None:
        s = ReplSession()
        s.eval_entry("input count: int")
        s.set_input("count", "42")
        r = s.eval_entry("count + 1")
        assert r.ok
        assert _int(r.value) == 43

    def test_set_undeclared_input_errors(self) -> None:
        s = ReplSession()
        with pytest.raises(AglError):
            s.set_input("nope", "1")

    def test_set_input_conversion_failure_errors(self) -> None:
        s = ReplSession()
        s.eval_entry("input count: int")
        with pytest.raises(AglError):
            s.set_input("count", "not-an-int")

    def test_set_input_excluded_from_bindings_until_set(self) -> None:
        s = ReplSession()
        s.eval_entry("input name: text")
        # unset input has no value, so it is not in bindings()
        assert all(n != "name" for n, _t, _v in s.bindings())
        s.set_input("name", "hi")
        assert any(n == "name" for n, _t, _v in s.bindings())


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        s.eval_entry("input n: int")
        s.reset()
        assert s.bindings() == []
        assert s.inputs() == []
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
        assert _int(r.value) == 2


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
            "var label: text = \"\"\n"
            "if n = 1 =>\n"
            "    set label = \"one\"\n"
            "| else =>\n"
            "    set label = \"many\"\n"
            "label\n"
        )
        s = ReplSession()
        results = s.load_file(f)
        assert all(r.ok for r in results), [r.diagnostics for r in results if not r.ok]
        vals = {n: v for n, _t, v in s.bindings()}
        assert _text(vals["label"]) == "one"

    def test_load_file_record_block_slices_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "rec.agl"
        f.write_text(
            "record Point\n"
            "    x: int\n"
            "    y: int\n"
            "let p = Point(x: 1, y: 2)\n"
            "p.x\n"
        )
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
    def test_non_exhaustive_case_warning_surfaced(self) -> None:
        s = ReplSession()
        s.eval_entry("enum R\n  | Pass\n  | Fail")
        s.eval_entry("let r: R = Pass")
        r = s.eval_entry("case r of\n  | Pass => pass")
        assert r.ok  # warnings never fail an entry
        assert len(r.warnings) == 1
        assert "Fail" in r.warnings[0].message

    def test_tab_warning_surfaced(self) -> None:
        # A TAB character in the entry source surfaces a per-entry advisory
        # warning (mirroring ``WorkflowRuntime.run``), without failing the entry.
        s = ReplSession()
        r = s.eval_entry("let x =\t1")
        assert r.ok
        assert any("TAB" in w.message or "tab" in w.message for w in r.warnings)

    def test_warning_on_check_only_path(self) -> None:
        s = ReplSession()
        s.eval_entry("enum R\n  | Pass\n  | Fail")
        s.eval_entry("let r: R = Pass")
        r = s.eval_entry("case r of\n  | Pass => pass", check_only=True)
        assert r.ok
        assert len(r.warnings) == 1


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
        assert not s.eval_entry("let p = P(x: 1)").ok

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

    def test_register_codec_and_renderer_share_validation(self) -> None:
        from agm.agl.eval.values import Value
        from agm.agl.runtime.codec import JsonCodec

        s = ReplSession()
        with pytest.raises(ValueError):
            s.register_codec(JsonCodec())  # reserved built-in name

        def my_renderer(value: Value, opt: str | None) -> str:
            return "x"

        with pytest.raises(ValueError):
            s.register_renderer("default", my_renderer)  # reserved name

    def test_registered_renderer_is_used(self) -> None:
        from agm.agl.eval.values import Value

        def shout(value: Value, opt: str | None) -> str:
            return "SHOUT"

        s = ReplSession()
        s.register_renderer("shout", shout)
        # Use it in an interpolation; the rendered output is consumed by print
        # (no observable echo in M1b) but the entry must type-check & run.
        s.eval_entry('let x = "hi"')
        r = s.eval_entry('print "${x as shout}"')
        assert r.ok


# ---------------------------------------------------------------------------
# EntryResult shape
# ---------------------------------------------------------------------------


class TestContractError:
    def test_contract_materialization_error_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object) -> object:
            raise ValueError("bad contract")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)
        s = ReplSession(default_agent=CountingAgent("ok"))
        r = s.eval_entry('let x = ask """hi"""')
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
# Agent-call cancellation (declined / interrupted) — atomic entry abort
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
        # Atomic: the failed entry promoted nothing.
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

    def test_cancellation_rolls_back_prior_set_mutation(self) -> None:
        # A ``set`` to a prior binding before a cancelled agent call must roll
        # back — the entry is atomic.
        s = ReplSession(default_agent=_CancellingAgent())
        s.eval_entry("var v = 1")
        r = s.eval_entry('do\n  set v = 2\n  let g = ask """x"""')
        assert not r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["v"] == 1  # the set was rolled back


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


# ---------------------------------------------------------------------------
# preset_input — the --input pre-seed flow
# ---------------------------------------------------------------------------


class TestPresetInput:
    def test_preset_applied_on_later_declaration(self) -> None:
        s = ReplSession()
        s.preset_input("count", "42")
        # Not declared yet → pending; not yet in inputs().
        assert s.inputs() == []
        s.eval_entry("input count: int")
        # Declaring the input applies the pending value.
        r = s.eval_entry("count + 1")
        assert r.ok
        assert _int(r.value) == 43

    def test_preset_for_already_declared_input_applies_immediately(self) -> None:
        s = ReplSession()
        s.eval_entry("input name: text")
        s.preset_input("name", "World")
        r = s.eval_entry("name")
        assert r.ok
        assert _text(r.value) == "World"

    def test_preset_bad_value_leaves_input_unset(self) -> None:
        s = ReplSession()
        s.preset_input("count", "not-an-int")
        s.eval_entry("input count: int")
        # Conversion failed → input stays unset; referencing it is a clean error.
        _name, _typ, val = s.inputs()[0]
        assert val is None
        r = s.eval_entry("count")
        assert not r.ok
        assert ":set" in r.diagnostics[0].message

    def test_preset_bad_value_for_declared_input_leaves_unset(self) -> None:
        s = ReplSession()
        s.eval_entry("input count: int")
        s.preset_input("count", "nope")  # swallowed, not raised
        _name, _typ, val = s.inputs()[0]
        assert val is None

    def test_reset_clears_pending_presets(self) -> None:
        s = ReplSession()
        s.preset_input("count", "42")
        s.reset()
        # After reset the pending value is gone: declaring leaves it unset.
        s.eval_entry("input count: int")
        _name, _typ, val = s.inputs()[0]
        assert val is None


# ---------------------------------------------------------------------------
# Issue #1 — re-declared input: stale value must be purged from value scope
# ---------------------------------------------------------------------------


class TestInputRedeclaration:
    def test_redeclare_input_purges_stale_value_from_bindings(self) -> None:
        """Re-declaring an already-:set input must remove the old value.

        After `input x: int` → `:set x=5` → `input x: int` again:
        - `inputs()` must report x as unset (value is None)
        - `bindings()` must NOT list x (no value in scope)
        The two tables must agree: no stale value survives.
        """
        s = ReplSession()
        r1 = s.eval_entry("input x: int")
        assert r1.ok
        s.set_input("x", "5")

        # Confirm x is set in both tables before re-declaring.
        ins = {name: val for name, _t, val in s.inputs()}
        assert ins["x"] is not None
        assert any(n == "x" for n, _t, _v in s.bindings())

        # Re-declare the same input.
        r2 = s.eval_entry("input x: int")
        assert r2.ok

        # inputs() must report x as unset.
        ins2 = {name: val for name, _t, val in s.inputs()}
        assert ins2["x"] is None, "Re-declared input must be unset in inputs()"

        # bindings() must NOT list x — the stale value must be gone from the
        # value scope so the two tables agree.
        assert all(n != "x" for n, _t, _v in s.bindings()), (
            "Re-declared input must not appear in bindings() with its stale value"
        )

    def test_redeclare_input_then_reference_raises_unset_guard(self) -> None:
        """After re-declaration the unset-input guard must fire on reference."""
        s = ReplSession()
        s.eval_entry("input x: int")
        s.set_input("x", "5")
        # Re-declare — x is now unset again.
        s.eval_entry("input x: int")
        r = s.eval_entry("x + 1")
        assert not r.ok
        assert r.diagnostics
        assert ":set" in r.diagnostics[0].message

    def test_redeclare_input_then_reset_works(self) -> None:
        """Re-set after re-declaration succeeds and value is usable."""
        s = ReplSession()
        s.eval_entry("input x: int")
        s.set_input("x", "5")
        s.eval_entry("input x: int")
        s.set_input("x", "10")
        r = s.eval_entry("x + 1")
        assert r.ok
        assert _int(r.value) == 11


# ---------------------------------------------------------------------------
# Issue #7 — snapshot optimisation: set-to-prior binding still rolls back
# ---------------------------------------------------------------------------


class TestSnapshotOptimisation:
    def test_set_to_prior_binding_in_raising_entry_rolls_back(self) -> None:
        """A ``set`` to a prior session binding that raises mid-entry rolls back.

        This verifies the rollback invariant is intact even when the snapshot
        optimisation narrows what is snapshotted.
        """
        s = ReplSession()
        r1 = s.eval_entry("var counter = 0")
        assert r1.ok
        # This entry sets counter=99 then raises (division by zero).
        r2 = s.eval_entry("set counter = 99\nlet _z: decimal = 1 / 0")
        assert not r2.ok
        assert r2.error is not None
        # The set must have been rolled back.
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["counter"] == 0

    def test_entry_without_set_does_not_corrupt_prior_bindings(self) -> None:
        """An entry with no ``set`` statements leaves prior bindings untouched.

        This guards that the optimisation (no snapshot for no-set entries) does
        not accidentally allow prior bindings to be mutated on success.
        """
        s = ReplSession()
        s.eval_entry("var a = 1")
        s.eval_entry("let b = 2")
        # An entry that only reads a and b, with no set.
        r = s.eval_entry("a + b")
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals["a"] == 1
        assert vals["b"] == 2

    def test_entry_with_only_new_bindings_does_not_disturb_prior(self) -> None:
        """Adding new bindings in an entry that raises leaves old bindings clean."""
        s = ReplSession()
        s.eval_entry("let x = 10")
        # Entry raises; it tries to add a new binding (no set to prior).
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

    def test_bare_if_stmt_classified_as_statement(self) -> None:
        # A bare if-statement at the prompt reduces to IfStmt (not ExprStmt),
        # so _classify returns "statement" — no value is echoed.
        # This mirrors the existing bare-case-statement behaviour.
        s = ReplSession()
        s.eval_entry("var x = 0")
        r = s.eval_entry("if true =>\n    set x = 42\n| else =>\n    set x = 0")
        assert r.ok
        assert r.kind == "statement"
        assert r.value is None
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
# helpers
# ---------------------------------------------------------------------------


def _int(value: object) -> int:
    from agm.agl.eval.values import IntValue

    assert isinstance(value, IntValue)
    return value.value


def _text(value: object) -> str:
    from agm.agl.eval.values import TextValue

    assert isinstance(value, TextValue)
    return value.value


def _snapshot(s: ReplSession) -> list[tuple[str, str, str]]:
    """A comparable snapshot of promoted bindings (name, type repr, value repr)."""
    return [(n, repr(t), repr(v)) for n, t, v in s.bindings()]
