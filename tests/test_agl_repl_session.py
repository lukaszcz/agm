"""Tests for the UI-free REPL session core (``agm.agl.repl.session``).

Drives ``ReplSession`` directly with source strings and fake agents.  Asserts
user-visible behaviour: persistence across entries, redefinition/shadowing,
expression/binding echo data, ``type_of`` purity, atomic-on-error promotion,
exactly-once agent dispatch, the ``:set`` input flow, ``reset``, ``load_file``,
``dump_source``, surfaced warnings, and ``check_only`` (type-only) runs.
"""

from __future__ import annotations

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
        assert s.type_of('prompt """ask"""') == repr(TextType())
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
        r1 = s.eval_entry('let g = prompt """say something"""')
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
        s.eval_entry('let a = prompt """q1"""')
        s.eval_entry('let b = prompt """q2"""')
        s.eval_entry('let c = prompt """q3"""')
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
        r = s.load_file(f)
        assert r.ok
        vals = {n: _int(v) for n, _t, v in s.bindings()}
        assert vals == {"a": 1, "b": 3}

    def test_load_file_agent_runs_once(self, tmp_path: Path) -> None:
        agent = CountingAgent("loaded")
        f = tmp_path / "p.agl"
        f.write_text('let g = prompt """hi"""\n')
        s = ReplSession(default_agent=agent)
        s.load_file(f)
        assert agent.calls == 1
        # Referencing it later does not re-run.
        s.eval_entry("g")
        assert agent.calls == 1

    def test_load_file_atomic_on_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.agl"
        f.write_text("let a = 1\nlet z: decimal = 1 / 0\n")
        s = ReplSession()
        r = s.load_file(f)
        assert not r.ok
        # Whole entry is atomic: 'a' (declared before the raise) is NOT promoted.
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
        r = s.eval_entry('prompt """ask"""', check_only=True)
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
    def test_agents_lists_named_and_prompt(self) -> None:
        s = ReplSession(default_agent=CountingAgent("x"))
        s.register_agent("alpha", CountingAgent("a"))
        s.register_agent("beta", CountingAgent("b"))
        assert s.agents() == ["alpha", "beta", "prompt"]

    def test_agents_without_default_excludes_prompt(self) -> None:
        s = ReplSession()
        s.register_agent("only", CountingAgent("x"))
        assert s.agents() == ["only"]

    def test_register_agent_reserved_name_rejected(self) -> None:
        s = ReplSession()
        with pytest.raises(ValueError):
            s.register_agent("prompt", CountingAgent("x"))

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
        r = s.eval_entry('let x = prompt """hi"""')
        assert not r.ok
        assert any("Contract error" in d.message for d in r.diagnostics)
        # Atomic: nothing promoted.
        assert s.bindings() == []


class TestEntryResultShape:
    def test_result_is_frozen_dataclass(self) -> None:
        s = ReplSession()
        r = s.eval_entry("let x = 1")
        assert isinstance(r, EntryResult)
        assert r.trace_path is None  # M1b: tracing is a no-op
        with pytest.raises(Exception):
            r.ok = False  # type: ignore[misc]


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
