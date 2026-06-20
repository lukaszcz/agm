"""Unit tests for the REPL renderer and meta-command dispatcher.

These are pure-data modules (no terminal), so they are tested directly:

- :func:`agm.agl.repl.render.render_entry_result` for each entry kind, warnings,
  pre-execution diagnostics, and runtime-error rendering;
- :func:`agm.agl.repl.meta.dispatch_meta` for ``:help`` / ``:quit`` / ``:exit`` /
  unknown commands, plus the ``register_meta_command`` extension hook used by M3.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.diagnostics import Diagnostic
from agm.agl.eval.values import IntValue, TextValue, Value
from agm.agl.repl import meta as meta_mod
from agm.agl.repl import render as render_mod
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.session import EntryKind, EntryResult, ReplSession
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.runtime.runtime import RunError
from agm.agl.typecheck.types import IntType, TextType, Type


class _CountingAgent:
    """Fake ``AgentFn`` returning scripted replies and counting invocations."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or ["ok"]
        self._next = 0
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        del request
        self.calls += 1
        reply = self._replies[min(self._next, len(self._replies) - 1)]
        self._next += 1
        return AgentResponse(content=reply)


def _result(
    *,
    kind: EntryKind = "statement",
    name: str | None = None,
    value: Value | None = None,
    value_type: Type | None = None,
    diagnostics: list[Diagnostic] | None = None,
    warnings: list[Diagnostic] | None = None,
    error: RunError | None = None,
    ok: bool = True,
) -> EntryResult:
    """Build an ``EntryResult`` with sensible defaults for one test axis."""
    return EntryResult(
        kind=kind,
        name=name,
        value=value,
        value_type=value_type,
        diagnostics=diagnostics if diagnostics is not None else [],
        warnings=warnings if warnings is not None else [],
        error=error,
        ok=ok,
    )


# ---------------------------------------------------------------------------
# render_entry_result
# ---------------------------------------------------------------------------


class TestRenderEntryResult:
    def test_expression_echo(self) -> None:
        result = _result(
            kind="expression", value=IntValue(Decimal(3)), value_type=IntType(), ok=True
        )
        assert render_mod.render_entry_result(result, echo=True) == "3"

    def test_expression_echo_quotes_text(self) -> None:
        # Strings are shown quoted in the REPL echo (interpolation is unaffected).
        result = _result(
            kind="expression", value=TextValue("aaa"), value_type=TextType(), ok=True
        )
        assert render_mod.render_entry_result(result, echo=True) == '"aaa"'

    def test_binding_echo_quotes_text(self) -> None:
        result = _result(
            kind="binding",
            name="g",
            value=TextValue("hi"),
            value_type=TextType(),
            ok=True,
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == 'g : text = "hi"'

    def test_binding_echo(self) -> None:
        result = _result(
            kind="binding",
            name="x",
            value=IntValue(Decimal(5)),
            value_type=IntType(),
            ok=True,
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "x : int = 5"

    def test_declaration_echo(self) -> None:
        result = _result(kind="declaration", name="R", ok=True)
        assert render_mod.render_entry_result(result, echo=True) == "R declared"

    def test_statement_echo_is_none(self) -> None:
        result = _result(kind="statement", ok=True)
        assert render_mod.render_entry_result(result, echo=True) is None

    def test_check_only_expression_shows_type(self) -> None:
        # In dry-run there is no value; the echo shows the inferred type.
        result = _result(kind="expression", value_type=IntType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == ": int"

    def test_check_only_binding_shows_name_and_type(self) -> None:
        result = _result(kind="binding", name="x", value_type=IntType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == "x : int"

    def test_check_only_declaration_confirms_name(self) -> None:
        result = _result(kind="declaration", name="R", ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == "R declared"

    def test_check_only_statement_is_none(self) -> None:
        result = _result(kind="statement", ok=True)
        assert render_mod.render_entry_result(result, echo=True, check_only=True) is None

    def test_check_only_echo_off_suppresses(self) -> None:
        result = _result(kind="expression", value_type=IntType(), ok=True)
        assert render_mod.render_entry_result(result, echo=False, check_only=True) is None

    def test_echo_off_suppresses_success(self) -> None:
        result = _result(
            kind="expression", value=TextValue("hi"), value_type=TextType(), ok=True
        )
        assert render_mod.render_entry_result(result, echo=False) is None

    def test_warnings_are_always_rendered(self) -> None:
        result = _result(
            kind="statement",
            ok=True,
            warnings=[
                Diagnostic(
                    message="watch out",
                    line=2,
                    column=5,
                    end_line=2,
                    end_column=10,
                    severity="warning",
                )
            ],
        )
        rendered = render_mod.render_entry_result(result, echo=False)
        assert rendered == "2:5-9: warning: watch out"

    def test_pre_execution_diagnostics(self) -> None:
        result = _result(
            ok=False,
            diagnostics=[Diagnostic(message="boom", line=1, column=4)],
        )
        assert render_mod.render_entry_result(result, echo=True) == "1:4: error: boom"

    def test_runtime_error_with_location(self) -> None:
        result = _result(
            ok=False,
            error=RunError(
                type_name="MyError",
                fields={"message": "bad"},
                line=4,
                col=7,
            ),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError: bad: at line 4, col 7"

    def test_runtime_error_line_only(self) -> None:
        result = _result(
            ok=False,
            error=RunError(type_name="MyError", fields={}, line=4, col=None),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError: at line 4"

    def test_runtime_error_no_location(self) -> None:
        result = _result(
            ok=False,
            error=RunError(type_name="MyError", fields={}, line=None, col=None),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError"

    def test_warning_then_error(self) -> None:
        result = _result(
            ok=False,
            warnings=[Diagnostic(message="w", line=1, column=1, severity="warning")],
            diagnostics=[Diagnostic(message="e", line=2, column=3)],
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "1:1: warning: w\n2:3: error: e"


# ---------------------------------------------------------------------------
# dispatch_meta
# ---------------------------------------------------------------------------


def _ctx() -> meta_mod.MetaContext:
    return meta_mod.MetaContext(session=ReplSession())


class TestDispatchMeta:
    def test_help_lists_commands(self) -> None:
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":help" in outcome.text
        assert ":quit" in outcome.text
        assert outcome.quit is False

    def test_quit(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":quit", ctx)
        assert outcome.quit is True
        assert ctx.quit is True

    def test_exit_alias(self) -> None:
        outcome = meta_mod.dispatch_meta(":exit", _ctx())
        assert outcome.quit is True

    def test_unknown_command(self) -> None:
        outcome = meta_mod.dispatch_meta(":nope", _ctx())
        assert outcome.text is not None
        assert "Unknown command ':nope'" in outcome.text
        assert outcome.quit is False

    def test_argument_is_ignored_for_known_command(self) -> None:
        # Trailing text after the command word is passed to the handler; :help
        # ignores it and still prints the list.
        outcome = meta_mod.dispatch_meta(":help everything", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text

    def test_meta_command_names_includes_aliases(self) -> None:
        names = meta_mod.meta_command_names()
        assert ":help" in names
        assert ":quit" in names
        assert ":exit" in names

    def test_register_meta_command_extends_registry(self) -> None:
        # M3 extension hook: a newly registered command becomes dispatchable and
        # is offered by ``meta_command_names`` (the completer's source).
        seen: list[str] = []

        def handler(arg: str, ctx: meta_mod.MetaContext) -> meta_mod.MetaOutcome:
            seen.append(arg)
            return meta_mod.MetaOutcome(text="did thing")

        command = meta_mod.MetaCommand(
            names=("xtest",),
            usage=":xtest",
            summary="A test command.",
            handler=handler,
        )
        meta_mod.register_meta_command(command)
        try:
            outcome = meta_mod.dispatch_meta(":xtest arg1", _ctx())
            assert outcome.text == "did thing"
            assert seen == ["arg1"]
            assert ":xtest" in meta_mod.meta_command_names()
        finally:
            meta_mod._COMMANDS.remove(command)

    def test_dispatch_tab_separated_command(self) -> None:
        # Issue #3: tab (or other whitespace) between command word and arg must
        # dispatch correctly, not produce "Unknown command".
        outcome = meta_mod.dispatch_meta(":type\tx + 1", _ctx())
        # ":type" with no valid binding → some error message, but NOT "Unknown command"
        assert outcome.text is not None
        assert "Unknown command" not in outcome.text

    def test_dispatch_tab_command_no_arg(self) -> None:
        # A tab immediately after a known command name with no trailing arg.
        outcome = meta_mod.dispatch_meta(":help\t", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text

    def test_command_index_cold_cache(self) -> None:
        # _command_index() must rebuild if its cache is None (cold-start path).
        original = meta_mod._command_index_cache
        meta_mod._command_index_cache = None
        try:
            idx = meta_mod._command_index()
            assert ":help" in idx or "help" in idx
        finally:
            meta_mod._command_index_cache = original

    def test_register_meta_command_cache_invalidation(self) -> None:
        # Issue #6: after registering a new command, dispatch and meta_command_names
        # must reflect the new entry (cache must be invalidated).
        seen: list[str] = []

        def handler2(arg: str, ctx: meta_mod.MetaContext) -> meta_mod.MetaOutcome:
            seen.append(arg)
            return meta_mod.MetaOutcome(text="cache-test")

        command2 = meta_mod.MetaCommand(
            names=("xcachetest",),
            usage=":xcachetest",
            summary="Cache invalidation test.",
            handler=handler2,
        )
        meta_mod.register_meta_command(command2)
        try:
            # Must be dispatchable immediately (cache invalidated on register).
            outcome = meta_mod.dispatch_meta(":xcachetest hello", _ctx())
            assert outcome.text == "cache-test"
            assert seen == ["hello"]
            # Must appear in names immediately.
            assert ":xcachetest" in meta_mod.meta_command_names()
        finally:
            meta_mod._COMMANDS.remove(command2)
            # Force cache rebuild by resetting (implementation detail: the cache
            # must reflect removal too — call meta_command_names after removal).

    def test_command_index_lazy_build_on_cold_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Verify the lazy-build branch of _command_index() is reachable: when both
        # caches are forced to None the functions rebuild them on the next call.
        monkeypatch.setattr(meta_mod, "_command_index_cache", None)
        monkeypatch.setattr(meta_mod, "_command_names_cache", None)
        # dispatch_meta → _command_index() triggers the lazy build of the index.
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text
        # meta_command_names() rebuilds the names cache when it is cold.
        monkeypatch.setattr(meta_mod, "_command_names_cache", None)
        names = meta_mod.meta_command_names()
        assert ":help" in names


# ---------------------------------------------------------------------------
# :set only controls REPL options
# ---------------------------------------------------------------------------


class TestSetOptions:
    def test_set_non_option_gives_usage(self) -> None:
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(":set count=5", _session_ctx(s))
        assert outcome.text is not None
        assert "usage" in outcome.text.lower()

    def test_set_unknown_form_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":set foo", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_set_empty_assignment_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":set =5", _session_ctx())
        assert "usage" in (outcome.text or "").lower()


# ---------------------------------------------------------------------------
# render helpers (single-sourced binding/value formatting)
# ---------------------------------------------------------------------------


class TestRenderHelpers:
    def test_format_typed_value(self) -> None:
        line = render_mod.format_typed_value("x", IntType(), IntValue(Decimal(5)))
        assert line == "x : int = 5"

    def test_binding_echo_matches_format_helper(self) -> None:
        # The entry-echo path and the shared helper must produce the same line.
        value = TextValue("hi")
        result = _result(kind="binding", name="g", value=value, value_type=TextType())
        echoed = render_mod.render_entry_result(result, echo=True)
        assert echoed == render_mod.format_typed_value("g", TextType(), value)


# ---------------------------------------------------------------------------
# Full meta-command set
# ---------------------------------------------------------------------------


def _session_ctx(
    session: ReplSession | None = None,
    *,
    agent_mode: AgentMode | None = None,
) -> meta_mod.MetaContext:
    return meta_mod.MetaContext(
        session=session if session is not None else ReplSession(),
        agent_mode=agent_mode if agent_mode is not None else AgentMode(),
    )


class TestHelpFullSet:
    def test_help_lists_full_command_set(self) -> None:
        out = meta_mod.dispatch_meta(":help", _session_ctx()).text
        assert out is not None
        for cmd in (":reset", ":type", ":bindings", ":env", ":agents", ":params",
                    ":set", ":agent", ":load", ":save"):
            assert cmd in out


class TestReset:
    def test_reset_clears_bindings(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        assert s.bindings()
        outcome = meta_mod.dispatch_meta(":reset", _session_ctx(s))
        assert "reset" in (outcome.text or "").lower()
        assert s.bindings() == []


class TestType:
    def test_type_of_valid_expr(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 5")
        outcome = meta_mod.dispatch_meta(":type x + 1", _session_ctx(s))
        assert outcome.text == "int"

    def test_type_empty_arg_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":type", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_type_unknown_name_clean_error(self) -> None:
        outcome = meta_mod.dispatch_meta(":type nope", _session_ctx())
        assert outcome.text is not None
        assert outcome.quit is False  # never crashed the loop

    def test_type_bad_syntax_clean_error(self) -> None:
        outcome = meta_mod.dispatch_meta(":type 1 +", _session_ctx())
        assert outcome.text is not None

    def test_type_non_expression_clean_error(self) -> None:
        # A binding is not a single expression — type_of raises AglError, caught.
        outcome = meta_mod.dispatch_meta(":type let y = 1", _session_ctx())
        assert outcome.text is not None
        assert "expression" in outcome.text.lower()


class TestBindings:
    def test_bindings_empty(self) -> None:
        for name in (":bindings", ":env"):
            outcome = meta_mod.dispatch_meta(name, _session_ctx())
            assert outcome.text == "No bindings."

    def test_bindings_lists_with_types_and_values(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 5")
        s.eval_entry('let g = "hi"')
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert "x : int = 5" in outcome.text
        assert 'g : text = "hi"' in outcome.text

    def test_env_alias_same_as_bindings(self) -> None:
        s = ReplSession()
        s.eval_entry("let x = 5")
        out_b = meta_mod.dispatch_meta(":bindings", _session_ctx(s)).text
        out_e = meta_mod.dispatch_meta(":env", _session_ctx(s)).text
        assert out_b == out_e


class TestAgents:
    def test_agents_empty_notes_default(self) -> None:
        outcome = meta_mod.dispatch_meta(":agents", _session_ctx())
        assert outcome.text is not None
        assert "mode: confirm" in outcome.text

    def test_agents_lists_registered_and_default_ask(self) -> None:
        s = ReplSession(default_agent=_CountingAgent("x"))
        s.register_agent("reviewer", _CountingAgent("r"))
        outcome = meta_mod.dispatch_meta(":agents", _session_ctx(s))
        assert outcome.text is not None
        assert "reviewer" in outcome.text
        assert "ask" in outcome.text

    def test_agents_reports_current_mode(self) -> None:
        mode = AgentMode(mode="auto")
        outcome = meta_mod.dispatch_meta(":agents", _session_ctx(agent_mode=mode))
        assert outcome.text is not None
        assert "mode: auto" in outcome.text


class TestInputs:
    def test_inputs_empty(self) -> None:
        outcome = meta_mod.dispatch_meta(":params", _session_ctx())
        assert outcome.text == "No params declared."

    def test_inputs_shows_unset_then_set(self) -> None:
        s = ReplSession()
        s.eval_entry('param name: text = "World"')
        out_set = meta_mod.dispatch_meta(":params", _session_ctx(s)).text
        assert out_set is not None
        assert 'name : text = "World"' in out_set


class TestSet:
    def test_set_declared_input(self) -> None:
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(":set count=42", _session_ctx(s))
        assert "usage" in (outcome.text or "").lower()

    def test_set_undeclared_input_clean_error(self) -> None:
        outcome = meta_mod.dispatch_meta(":set nope=1", _session_ctx())
        assert outcome.text is not None
        assert "usage" in outcome.text.lower()

    def test_set_bad_value_clean_error(self) -> None:
        s = ReplSession()
        s.eval_entry("param count: int")
        outcome = meta_mod.dispatch_meta(":set count=oops", _session_ctx(s))
        assert outcome.text is not None

    def test_set_missing_equals_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":set foo", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_set_empty_name_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":set =5", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_set_echo_off_then_on_toggles_ctx(self) -> None:
        ctx = _session_ctx()
        assert ctx.echo is True
        off = meta_mod.dispatch_meta(":set echo off", ctx)
        assert ctx.echo is False
        assert "off" in (off.text or "").lower()
        on = meta_mod.dispatch_meta(":set echo on", ctx)
        assert ctx.echo is True
        assert "on" in (on.text or "").lower()

    def test_set_echo_bad_state_gives_usage(self) -> None:
        ctx = _session_ctx()
        outcome = meta_mod.dispatch_meta(":set echo maybe", ctx)
        assert "usage" in (outcome.text or "").lower()
        assert ctx.echo is True  # unchanged


class TestAgent:
    def test_agent_auto_then_confirm_mutates_shared_mode(self) -> None:
        mode = AgentMode()
        ctx = _session_ctx(agent_mode=mode)
        out_auto = meta_mod.dispatch_meta(":agent auto", ctx)
        assert mode.mode == "auto"
        assert "auto" in (out_auto.text or "")
        out_conf = meta_mod.dispatch_meta(":agent confirm", ctx)
        assert mode.mode == "confirm"
        assert "confirm" in (out_conf.text or "")

    def test_agent_no_arg_reports_mode(self) -> None:
        mode = AgentMode(mode="auto")
        outcome = meta_mod.dispatch_meta(":agent", _session_ctx(agent_mode=mode))
        assert "auto" in (outcome.text or "")

    def test_agent_bad_arg_usage_error_no_mutation(self) -> None:
        mode = AgentMode()
        outcome = meta_mod.dispatch_meta(":agent bogus", _session_ctx(agent_mode=mode))
        assert "usage" in (outcome.text or "").lower()
        assert mode.mode == "confirm"


class TestLoad:
    def test_load_runs_file_into_session(self, tmp_path: Path) -> None:
        src = tmp_path / "prog.agl"
        src.write_text("let x = 7\n")
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "x : int = 7" in outcome.text
        # The binding persisted into the session.
        assert any(n == "x" for n, _t, _v in s.bindings())

    def test_load_agent_call_fires_exactly_once(self, tmp_path: Path) -> None:
        src = tmp_path / "agent.agl"
        src.write_text('let r = ask """do it"""\n')
        agent = _CountingAgent("done")
        s = ReplSession(default_agent=agent)
        meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert agent.calls == 1

    def test_load_missing_file_clean_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.agl"
        outcome = meta_mod.dispatch_meta(f":load {missing}", _session_ctx())
        assert outcome.text is not None
        assert "cannot read" in outcome.text.lower()

    def test_load_empty_arg_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":load", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_load_renders_each_statement(self, tmp_path: Path) -> None:
        # Multiple statements load incrementally; each statement's echo surfaces.
        src = tmp_path / "multi.agl"
        src.write_text("let a = 1\nlet b = 2\n")
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "a : int = 1" in outcome.text
        assert "b : int = 2" in outcome.text

    def test_load_halts_at_first_error(self, tmp_path: Path) -> None:
        src = tmp_path / "halt.agl"
        src.write_text("let a = 1\nlet z: decimal = 1 / 0\nlet b = 99\n")
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        # The failing statement's error surfaced; the unreached one did not run.
        assert "exception" in outcome.text.lower()
        names = {n for n, _t, _v in s.bindings()}
        assert names == {"a"}

    def test_load_empty_file_benign_note(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.agl"
        src.write_text("# only a comment\n")
        s = ReplSession()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "no statements" in outcome.text.lower()
        assert s.bindings() == []


class TestSave:
    def test_save_round_trips_source(self, tmp_path: Path) -> None:
        s = ReplSession()
        s.eval_entry("let x = 1")
        s.eval_entry("let y = 2")
        out = tmp_path / "out.agl"
        outcome = meta_mod.dispatch_meta(f":save {out}", _session_ctx(s))
        assert str(out) in (outcome.text or "")
        assert out.read_text() == s.dump_source()
        # The saved source replays into a fresh session.
        s2 = ReplSession()
        assert all(r.ok for r in s2.load_file(out))

    def test_save_load_round_trips_redefinition(self, tmp_path: Path) -> None:
        # A transcript containing a redefinition must round-trip through
        # :save -> :load (each statement loads as its own entry, so the second
        # `let x` shadows rather than being a duplicate-declaration error).
        s = ReplSession()
        s.eval_entry("let x = 1")
        s.eval_entry("let x = 2")
        out = tmp_path / "redef.agl"
        meta_mod.dispatch_meta(f":save {out}", _session_ctx(s))

        s2 = ReplSession()
        outcome = meta_mod.dispatch_meta(f":load {out}", _session_ctx(s2))
        # No error surfaced and x reloaded as the shadowed value 2.
        assert "line" not in (outcome.text or "")
        assert "x : int = 2" in (outcome.text or "")
        vals = {n: v for n, _t, v in s2.bindings()}
        assert isinstance(vals["x"], IntValue)
        assert vals["x"].value == 2

    def test_save_empty_arg_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":save", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_save_unwritable_path_clean_error(self, tmp_path: Path) -> None:
        # A path whose parent directory does not exist cannot be written.
        bad = tmp_path / "missing_dir" / "out.agl"
        outcome = meta_mod.dispatch_meta(f":save {bad}", _session_ctx())
        assert outcome.text is not None
        assert "cannot write" in outcome.text.lower()


class TestTheme:
    def test_no_arg_reports_current_theme(self) -> None:
        ctx = _ctx()
        ctx.theme = "dark"
        outcome = meta_mod.dispatch_meta(":theme", ctx)
        assert outcome.text is not None
        assert "dark" in outcome.text

    def test_switch_to_light(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":theme light", ctx)
        assert ctx.theme == "light"
        assert outcome.text is not None
        assert "light" in outcome.text

    def test_switch_to_dark(self) -> None:
        ctx = _ctx()
        ctx.theme = "light"
        meta_mod.dispatch_meta(":theme dark", ctx)
        assert ctx.theme == "dark"

    def test_switch_to_auto(self) -> None:
        ctx = _ctx()
        meta_mod.dispatch_meta(":theme auto", ctx)
        assert ctx.theme == "auto"

    def test_unknown_theme_returns_error(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":theme neon", ctx)
        assert outcome.text is not None
        assert "Unknown theme" in outcome.text
        assert ctx.theme == "auto"  # unchanged

    def test_theme_in_help(self) -> None:
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":theme" in outcome.text

    def test_theme_in_meta_command_names(self) -> None:
        assert ":theme" in meta_mod.meta_command_names()
