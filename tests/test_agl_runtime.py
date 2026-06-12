"""Tests for WorkflowRuntime — M0 shell behaviors preserved, M1 additions.

Covers:
- WorkflowRuntime constructor with default kwargs
- register_agent: duplicate rejection, reserved-name rejection (prompt, exec)
- run: full pipeline now active; valid programs succeed; static errors fail
- Diagnostic has .message (str) and .line (int)
- RunResult has .ok (bool), .diagnostics (list), .error (None for pre-exec failures)
- AglError and SourceSpan
- Token constants
- M1 additions: agent registration/fallback, capability derivation, input validation,
  text-codec behavior, AgentCallError, empty-response valid case
"""

from __future__ import annotations

import pytest

from agm.agl import AglError, SourceSpan, WorkflowRuntime
from agm.agl.runtime import AgentRequest
from agm.agl.runtime.runtime import Diagnostic, RunResult


class TestWorkflowRuntimeConstructor:
    def test_default_constructor_uses_documented_defaults(self) -> None:
        rt = WorkflowRuntime()
        # Documented constructor defaults (design §2.8/§2.11).
        assert rt.default_loop_limit == 5
        assert rt.default_strict_json is False

    def test_default_loop_limit_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(default_loop_limit=10)
        assert rt.default_loop_limit == 10

    def test_default_strict_json_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(default_strict_json=True)
        assert rt.default_strict_json is True

    def test_default_agent_constructed_runtime_runs(self) -> None:
        # A default_agent does not reserve the agent-name namespace: a runtime
        # built with one still accepts named registrations and still runs.
        def my_agent(request: object) -> str:
            return "response"

        rt = WorkflowRuntime(default_agent=my_agent)
        rt.register_agent("reviewer", my_agent)  # should not raise
        result = rt.run("let x = 1")
        # M1: a valid program with no agent calls returns ok=True
        assert result.ok is True
        assert result.error is None


class TestRegisterAgent:
    def test_register_agent_accepted(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        rt.register_agent("my_agent", my_agent)  # should not raise

    def test_register_duplicate_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        rt.register_agent("my_agent", my_agent)
        with pytest.raises(ValueError, match="my_agent"):
            rt.register_agent("my_agent", my_agent)

    def test_register_reserved_name_prompt_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="prompt"):
            rt.register_agent("prompt", my_agent)

    def test_register_reserved_name_exec_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="exec"):
            rt.register_agent("exec", my_agent)


class TestRunBehavior:
    """M1 run() behavior: valid programs run, static errors fail cleanly."""

    def test_run_returns_run_result(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert isinstance(result, RunResult)

    def test_valid_program_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is True

    def test_static_error_not_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) >= 1

    def test_static_error_diagnostic_has_message(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.message, str)
        assert diag.message

    def test_static_error_diagnostic_has_line(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.line, int)
        assert diag.line >= 1

    def test_run_result_error_none_for_static_failure(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        # pre-execution failure: error is None (no AgL exception was raised)
        assert result.error is None

    def test_run_with_inputs(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input k\nprint k", inputs={"k": "value"})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_with_empty_inputs(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1", inputs={})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_parse_error_not_ok(self) -> None:
        rt = WorkflowRuntime()
        # Invalid syntax
        result = rt.run("@@@@@")
        assert result.ok is False
        assert result.error is None


class TestFallbackAgent:
    """has_fallback_agent behavior for capability checking."""

    def test_no_default_agent_prompt_call_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default_agent
        result = rt.run('let x = prompt "hi"')
        assert result.ok is False
        assert result.error is None  # static, not runtime

    def test_with_default_agent_prompt_call_succeeds(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('let x = prompt "hi"')
        assert result.ok is True

    def test_named_agent_registered_accepted(self) -> None:
        rt = WorkflowRuntime()
        rt.register_agent("impl", lambda req: "output")
        result = rt.run('let x = impl "do it"')
        assert result.ok is True

    def test_unknown_named_agent_without_fallback_is_error(self) -> None:
        rt = WorkflowRuntime()
        # No agents registered, no fallback → static error for named agent
        result = rt.run('let x = mysterious_agent "hi"')
        assert result.ok is False
        assert result.error is None

    def test_has_fallback_when_default_agent_is_set(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # A runtime with a default_agent provides fallback for any name
        result = rt.run('let x = any_agent_name "hi"')
        assert result.ok is True


class TestInputValidationRuntime:
    """Input validation before execution (§11.3, §9.5)."""

    def test_missing_input_fails_not_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input spec\nprint spec", inputs={})
        assert result.ok is False
        assert result.error is None  # host error, not AgL exception

    def test_missing_input_mentions_name(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input spec\nprint spec", inputs={})
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "spec" in msgs.lower()

    def test_undeclared_extra_fails(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input a\nprint a", inputs={"a": "ok", "b": "extra"})
        assert result.ok is False
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "b" in msgs.lower()

    def test_text_input_verbatim(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input msg\nprint msg", inputs={"msg": "hello world"})
        assert result.ok is True

    def test_no_agent_called_on_input_failure(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run("input x\nlet y = prompt \"Hi\"", inputs={})
        assert calls == []

    def test_int_input_json_parsed(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input n: int\nprint n", inputs={"n": 5})
        assert result.ok
        out = capsys.readouterr().out
        assert "5" in out

    def test_invalid_typed_input_fails(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("input n: int\nprint n", inputs={"n": "five"})
        assert result.ok is False
        assert result.error is None


class TestEmptyResponse:
    """Exit 0 with empty stdout is a valid empty response (plan §9.5)."""

    def test_empty_string_response_is_valid_text(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "")
        result = rt.run('let x = prompt "Say nothing."')
        assert result.ok is True
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("")


class TestAgentRequest:
    """AgentRequest contract: .prompt and .agent fields."""

    def test_request_prompt_is_rendered_template(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('let x = prompt "Hello world"')
        assert received[0].prompt == "Hello world"

    def test_request_agent_name_for_default(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('let x = prompt "Hi"')
        assert received[0].agent == "prompt"

    def test_request_agent_name_for_named(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this"')
        assert received[0].agent == "reviewer"


class TestDiagnosticType:
    def test_diagnostic_attributes(self) -> None:
        d = Diagnostic(message="some error", line=3)
        assert d.message == "some error"
        assert d.line == 3

    def test_diagnostic_defaults_to_error_severity(self) -> None:
        d = Diagnostic(message="some error", line=3)
        assert d.severity == "error"

    def test_diagnostic_warning_severity(self) -> None:
        d = Diagnostic(message="a warning", line=2, severity="warning")
        assert d.severity == "warning"
        assert d.message == "a warning"
        assert d.line == 2


class TestRunResultType:
    def test_run_result_attributes(self) -> None:
        d = Diagnostic(message="err", line=1)
        result = RunResult(ok=False, diagnostics=[d], error=None)
        assert result.ok is False
        assert result.diagnostics == [d]
        assert result.error is None

    def test_run_result_ok_with_only_warnings(self) -> None:
        # ok may be True even when warning-severity diagnostics are present:
        # warnings are reported but do not make the run fail.
        warning = Diagnostic(message="exhaustiveness", line=4, severity="warning")
        result = RunResult(ok=True, diagnostics=[warning], error=None)
        assert result.ok is True
        assert result.diagnostics == [warning]
        assert result.error is None

    def test_run_result_has_bindings(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert hasattr(result, "bindings")
        assert isinstance(result.bindings, dict)


class TestSourceSpan:
    def test_source_span_attributes(self) -> None:
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=11,
            start_offset=0, end_offset=10,
        )
        assert span.start_line == 1
        assert span.start_col == 1
        assert span.end_line == 1
        assert span.end_col == 11


class TestAglError:
    def test_agl_error_no_span(self) -> None:
        err = AglError("something went wrong")
        diag = err.to_diagnostic()
        assert diag.line == 1
        assert "something went wrong" in diag.message

    def test_agl_error_with_span(self) -> None:
        span = SourceSpan(
            start_line=5, start_col=3, end_line=5, end_col=12,
            start_offset=40, end_offset=49,
        )
        err = AglError("type error", span=span)
        diag = err.to_diagnostic()
        assert diag.line == 5
        assert "type error" in diag.message

    def test_agl_error_span_attribute(self) -> None:
        span = SourceSpan(
            start_line=2, start_col=1, end_line=2, end_col=5,
            start_offset=10, end_offset=14,
        )
        err = AglError("test", span=span)
        assert err.span is span

    def test_agl_error_no_span_is_none(self) -> None:
        err = AglError("test")
        assert err.span is None


class TestTokenConstants:
    """Verify the token alphabet is importable and has the expected constants."""

    def test_layout_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import DEDENT, INDENT, NEWLINE

        assert NEWLINE == "_NEWLINE"
        assert INDENT == "_INDENT"
        assert DEDENT == "_DEDENT"

    def test_template_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import (
            INTERP_END,
            INTERP_START,
            STRING_FRAGMENT,
            TEMPLATE_END,
            TEMPLATE_START,
        )

        assert TEMPLATE_START == "TEMPLATE_START"
        assert STRING_FRAGMENT == "STRING_FRAGMENT"
        assert INTERP_START == "INTERP_START"
        assert INTERP_END == "INTERP_END"
        assert TEMPLATE_END == "TEMPLATE_END"

    def test_identifier_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import TYPE_NAME, VAR_NAME

        assert TYPE_NAME == "TYPE_NAME"
        assert VAR_NAME == "VAR_NAME"

    def test_number_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import DECIMAL, INT

        assert INT == "INT"
        assert DECIMAL == "DECIMAL"

    def test_keywords_frozenset_contains_expected(self) -> None:
        from agm.agl.lexer.tokens import KEYWORDS

        assert "let" in KEYWORDS
        assert "var" in KEYWORDS
        assert "if" in KEYWORDS
        assert "case" in KEYWORDS
        assert "do" in KEYWORDS
        assert "until" in KEYWORDS
        assert "try" in KEYWORDS
        assert "catch" in KEYWORDS
        assert "record" in KEYWORDS
        assert "enum" in KEYWORDS
        # contextual keywords NOT in KEYWORDS
        assert "prompt" not in KEYWORDS
        assert "exec" not in KEYWORDS

    def test_operators_defined(self) -> None:
        from agm.agl.lexer.tokens import (
            ARROW,
            COLON,
            COMMA,
            DOT,
            EQ,
            GE,
            GT,
            LE,
            LT,
            MINUS,
            NEQ,
            PIPE,
            PLUS,
            SEMICOLON,
            SLASH,
            STAR,
        )

        assert ARROW == "ARROW"
        assert EQ == "EQ"
        assert NEQ == "NEQ"
        assert LE == "LE"
        assert GE == "GE"
        assert LT == "LT"
        assert GT == "GT"
        assert PLUS == "PLUS"
        assert MINUS == "MINUS"
        assert STAR == "STAR"
        assert SLASH == "SLASH"
        assert COLON == "COLON"
        assert COMMA == "COMMA"
        assert DOT == "DOT"
        assert PIPE == "PIPE"
        assert SEMICOLON == "SEMICOLON"

    def test_bracket_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import LBRACE, LPAR, LSQB, RBRACE, RPAR, RSQB

        assert LPAR == "LPAR"
        assert RPAR == "RPAR"
        assert LSQB == "LSQB"
        assert RSQB == "RSQB"
        assert LBRACE == "LBRACE"
        assert RBRACE == "RBRACE"

    def test_error_token_defined(self) -> None:
        from agm.agl.lexer.tokens import EQ_EQ

        assert EQ_EQ == "EQ_EQ"

    def test_all_keyword_constants(self) -> None:
        from agm.agl.lexer import tokens

        assert tokens.KW_RECORD == "record"
        assert tokens.KW_ENUM == "enum"
        assert tokens.KW_TYPE == "type"
        assert tokens.KW_INPUT == "input"
        assert tokens.KW_LET == "let"
        assert tokens.KW_VAR == "var"
        assert tokens.KW_SET == "set"
        assert tokens.KW_DO == "do"
        assert tokens.KW_UNTIL == "until"
        assert tokens.KW_IF == "if"
        assert tokens.KW_ELSE == "else"
        assert tokens.KW_CASE == "case"
        assert tokens.KW_OF == "of"
        assert tokens.KW_TRY == "try"
        assert tokens.KW_CATCH == "catch"
        assert tokens.KW_RAISE == "raise"
        assert tokens.KW_AS == "as"
        assert tokens.KW_PASS == "pass"
        assert tokens.KW_PRINT == "print"
        assert tokens.KW_AND == "and"
        assert tokens.KW_OR == "or"
        assert tokens.KW_NOT == "not"
        assert tokens.KW_IS == "is"
        assert tokens.KW_IN == "in"
        assert tokens.KW_TRUE == "true"
        assert tokens.KW_FALSE == "false"
        assert tokens.KW_NULL == "null"


class TestWorkflowRuntimeProperties:
    def test_default_loop_limit_property(self) -> None:
        rt = WorkflowRuntime(default_loop_limit=7)
        assert rt.default_loop_limit == 7

    def test_default_strict_json_property(self) -> None:
        rt = WorkflowRuntime(default_strict_json=True)
        assert rt.default_strict_json is True

    def test_default_strict_json_property_false(self) -> None:
        rt = WorkflowRuntime(default_strict_json=False)
        assert rt.default_strict_json is False


class TestNoDefaultAgent:
    """F1a/F1b: a ``prompt`` call needs a default (or fallback) agent."""

    def test_prompt_without_default_agent_is_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default agent configured
        result = rt.run('let x = prompt "hi"')
        assert result.ok is False
        assert result.error is None  # static (pre-execution), not an AgL exception
        assert any("default agent" in d.message.lower() for d in result.diagnostics)

    def test_prompt_with_default_agent_runs(self) -> None:
        def agent(request: object) -> str:
            return "answer"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = prompt "hi"')
        assert result.ok is True
        assert result.error is None


class TestDryRunCheckOnly:
    """F2: ``check_only=True`` runs the static pipeline but executes nothing."""

    def test_check_only_printing_program_produces_no_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run('print "hello"', check_only=True)
        assert result.ok is True
        assert result.bindings == {}
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_check_only_static_error_still_fails(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name", check_only=True)
        assert result.ok is False

    def test_check_only_never_invokes_agent(self) -> None:
        calls: list[object] = []

        def agent(request: object) -> str:
            calls.append(request)
            return "should not be called"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = prompt "hi"', check_only=True)
        assert result.ok is True
        # The agent must never be invoked during a dry run.
        assert calls == []

    def test_check_only_input_validation_still_runs(self) -> None:
        rt = WorkflowRuntime()
        # Missing declared input is caught even under check_only.
        result = rt.run("input msg\nprint msg", inputs={}, check_only=True)
        assert result.ok is False
        assert any("msg" in d.message for d in result.diagnostics)


class TestDecimalSerialization:
    """F3/F9: decimals print/round-trip exactly; never via binary float."""

    def test_json_input_with_decimal_prints_exactly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run(
            'input data: json\nprint data', inputs={"data": '{"a": 1.5}'}
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "1.5" in captured.out
        # No binary-float artifacts (e.g. 1.5000000000000002).
        assert "1.5000" not in captured.out

    def test_decimal_value_prints_exact_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let x = 0.1\nprint x')
        assert result.ok is True
        captured = capsys.readouterr()
        assert captured.out.strip() == "0.1"

    def test_run_error_preserves_decimal_exactness(self) -> None:
        import decimal

        from agm.agl.eval.values import DecimalValue, ExceptionValue, TextValue
        from agm.agl.runtime.runtime import _exception_value_to_run_error

        exc = ExceptionValue(
            type_name="ValidationError",
            fields={
                "message": TextValue("bad"),
                "amount": DecimalValue(decimal.Decimal("0.1")),
            },
        )
        err = _exception_value_to_run_error(exc)
        assert err.fields["amount"] == decimal.Decimal("0.1")
        assert isinstance(err.fields["amount"], decimal.Decimal)


class TestWarningsThreadedOnFailurePaths:
    """F14: typecheck warnings survive input-validation failure paths."""

    def test_warning_and_missing_input_both_visible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M1 produces no checker warnings organically, so inject one through the
        # CheckedProgram the runtime threads from ``check``.  This exercises the
        # real failure path (missing input) while a warning is present.
        import agm.agl.typecheck as tc_mod
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.typecheck.env import CheckedProgram

        real_check = tc_mod.check
        warning = Diagnostic(message="a checker warning", line=1, severity="warning")

        def check_with_warning(resolved: object, caps: object) -> CheckedProgram:
            checked = real_check(resolved, caps)
            return CheckedProgram(
                resolved=checked.resolved,
                node_types=checked.node_types,
                contract_specs=checked.contract_specs,
                warnings=(*checked.warnings, warning),
                type_env=checked.type_env,
            )

        monkeypatch.setattr(tc_mod, "check", check_with_warning)

        rt = WorkflowRuntime()
        result = rt.run("input msg\nprint msg", inputs={})
        assert result.ok is False
        messages = [d.message for d in result.diagnostics]
        # Both the warning and the missing-input error are present.
        assert any("a checker warning" in m for m in messages)
        assert any("msg" in m for m in messages)


class TestAgentRegistryDispatch:
    """F17: dispatch resolves named agents, prompt, and the default fallback."""

    def test_dispatch_named_agent(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def named(req: AgentRequest) -> str:
            return f"named:{req.prompt}"

        registry = AgentRegistry(named={"reviewer": named}, default_agent=None)
        resp = registry.dispatch("reviewer", AgentRequest(agent="reviewer", prompt="hi"))
        assert resp.content == "named:hi"

    def test_dispatch_prompt_and_unknown_fall_back_to_default(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def default(req: AgentRequest) -> str:
            return f"default:{req.agent}"

        registry = AgentRegistry(named={}, default_agent=default)
        # Both ``prompt`` and an unregistered named agent route to the default.
        assert registry.dispatch("prompt", AgentRequest(agent="prompt", prompt="q")).content == (
            "default:prompt"
        )
        assert registry.dispatch("other", AgentRequest(agent="other", prompt="q")).content == (
            "default:other"
        )

    def test_dispatch_unknown_without_default_raises(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        registry = AgentRegistry(named={}, default_agent=None)
        with pytest.raises(KeyError, match="No agent registered"):
            registry.dispatch("ghost", AgentRequest(agent="ghost", prompt="q"))


class TestInputBindingInvariant:
    """The runtime relies on the checker recording every input's binding type."""

    def test_missing_binding_type_is_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.typecheck.env import TypeEnvironment

        # Force the checker invariant to be violated: no recorded binding type.
        monkeypatch.setattr(
            TypeEnvironment, "get_binding_type", lambda self, node_id: None
        )

        rt = WorkflowRuntime()
        with pytest.raises(AssertionError, match="binding type"):
            rt.run("input msg\nprint msg", inputs={"msg": "hi"})


class TestCollectAgentCallsWalker:
    """Unit tests for _collect_agent_calls: covers all AST node type branches.

    The walker must handle every expression and statement variant to support
    future-syntax constructs (M3/M4/M5) that are not yet parseable.  We build
    hand-crafted AST programs to exercise every branch.
    """

    def _sp(self) -> object:
        from agm.agl.syntax.spans import SourceSpan

        return SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=2, start_offset=0, end_offset=1
        )

    def _nid(self) -> int:
        import itertools

        if not hasattr(self, "_nid_counter"):
            self._nid_counter = itertools.count(10000)
        return next(self._nid_counter)

    def _call_options(self, policy: object = None) -> object:
        from agm.agl.syntax.nodes import CallOptions

        return CallOptions(
            format=None, strict_json=None, parse_policy=policy,
            span=self._sp(), node_id=self._nid(),
        )

    def _text_segment(self, text: str = "Q") -> object:
        from agm.agl.syntax.nodes import TextSegment

        return TextSegment(text=text, span=self._sp(), node_id=self._nid())

    def _template(self, *segs: object) -> object:
        from agm.agl.syntax.nodes import Template

        segments = tuple(segs) if segs else (self._text_segment(),)
        return Template(segments=segments, span=self._sp(), node_id=self._nid())

    def _agent_call(self, agent: str = "prompt", options: object = None) -> object:
        from agm.agl.syntax.nodes import AgentCall

        return AgentCall(
            agent=agent,
            options=options or self._call_options(),
            template=self._template(),
            span=self._sp(),
            node_id=self._nid(),
        )

    def _intlit(self) -> object:
        from agm.agl.syntax.nodes import IntLit

        return IntLit(value=0, span=self._sp(), node_id=self._nid())

    def _program(self, *stmts: object) -> object:
        from agm.agl.syntax.nodes import Program

        return Program(body=tuple(stmts), span=self._sp(), node_id=self._nid())

    def _wildcard_pattern(self) -> object:
        from agm.agl.syntax.nodes import WildcardPattern

        return WildcardPattern(span=self._sp(), node_id=self._nid())

    def _collect(self, program: object) -> list[object]:
        from agm.agl.runtime.runtime import _collect_agent_calls

        return _collect_agent_calls(program)

    def test_template_standalone_expr(self) -> None:
        """A bare Template expression in a LetDecl is traversed."""
        from agm.agl.syntax.nodes import LetDecl

        tmpl = self._template()
        let = LetDecl(name="x", type_ann=None, value=tmpl, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        result = self._collect(prog)
        assert result == []

    def test_interp_segment_with_nested_agent_call(self) -> None:
        """An AgentCall nested inside an InterpSegment is collected."""
        from agm.agl.syntax.nodes import InterpSegment, LetDecl, Template

        inner_call = self._agent_call()
        seg = InterpSegment(
            expr=inner_call, render=None, span=self._sp(), node_id=self._nid()
        )
        outer_tmpl = Template(segments=(seg,), span=self._sp(), node_id=self._nid())
        let = LetDecl(
            name="t", type_ann=None, value=outer_tmpl, span=self._sp(), node_id=self._nid()
        )
        prog = self._program(let)
        result = self._collect(prog)
        assert len(result) == 1
        assert result[0] is inner_call

    def test_binary_op_traverses_both_sides(self) -> None:
        """BinaryOp: both left and right sub-expressions are traversed."""
        from agm.agl.syntax.nodes import BinaryOp, LetDecl

        call_left = self._agent_call()
        call_right = self._agent_call()
        op = BinaryOp(
            op="==", left=call_left, right=call_right, span=self._sp(), node_id=self._nid()
        )
        let = LetDecl(name="r", type_ann=None, value=op, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        result = self._collect(prog)
        assert len(result) == 2

    def test_unary_not_traverses_operand(self) -> None:
        """UnaryNot: operand is traversed."""
        from agm.agl.syntax.nodes import LetDecl, UnaryNot

        call = self._agent_call()
        expr = UnaryNot(operand=call, span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_unary_neg_traverses_operand(self) -> None:
        """UnaryNeg: operand is traversed."""
        from agm.agl.syntax.nodes import LetDecl, UnaryNeg

        call = self._agent_call()
        expr = UnaryNeg(operand=call, span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_is_test_traverses_expr(self) -> None:
        """IsTest: the expression being tested is traversed."""
        from agm.agl.syntax.nodes import IsTest, LetDecl

        call = self._agent_call()
        expr = IsTest(
            expr=call, qualifier=None, variant="V", negated=False,
            span=self._sp(), node_id=self._nid(),
        )
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_case_expr_traverses_subject_and_branches(self) -> None:
        """CaseExpr: subject and each branch body are traversed."""
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, LetDecl

        call_subj = self._agent_call()
        call_body = self._agent_call()
        branch = CaseExprBranch(
            pattern=self._wildcard_pattern(), body=call_body,
            span=self._sp(), node_id=self._nid(),
        )
        expr = CaseExpr(subject=call_subj, branches=(branch,), span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        result = self._collect(prog)
        assert len(result) == 2

    def test_field_access_traverses_obj(self) -> None:
        """FieldAccess: the object expression is traversed."""
        from agm.agl.syntax.nodes import FieldAccess, LetDecl

        call = self._agent_call()
        expr = FieldAccess(obj=call, field="f", span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_constructor_traverses_arg_values(self) -> None:
        """Constructor: all named argument values are traversed."""
        from agm.agl.syntax.nodes import Constructor, LetDecl, NamedArg

        call = self._agent_call()
        arg = NamedArg(name="x", value=call, span=self._sp(), node_id=self._nid())
        expr = Constructor(
            qualifier=None, name="P", args=(arg,), span=self._sp(), node_id=self._nid()
        )
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_list_lit_traverses_elements(self) -> None:
        """ListLit: all element expressions are traversed."""
        from agm.agl.syntax.nodes import LetDecl, ListLit

        call = self._agent_call()
        expr = ListLit(elements=(call,), span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_dict_lit_traverses_entry_values(self) -> None:
        """DictLit: all entry values are traversed."""
        from agm.agl.syntax.nodes import DictEntry, DictLit, LetDecl

        call = self._agent_call()
        entry = DictEntry(key="k", value=call, span=self._sp(), node_id=self._nid())
        expr = DictLit(entries=(entry,), span=self._sp(), node_id=self._nid())
        let = LetDecl(name="r", type_ann=None, value=expr, span=self._sp(), node_id=self._nid())
        prog = self._program(let)
        assert len(self._collect(prog)) == 1

    def test_set_stmt_traverses_value(self) -> None:
        """SetStmt: the assigned value is traversed."""
        from agm.agl.syntax.nodes import SetStmt

        call = self._agent_call()
        stmt = SetStmt(target="x", value=call, span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 1

    def test_expr_stmt_traverses_expr(self) -> None:
        """ExprStmt: the expression is traversed."""
        from agm.agl.syntax.nodes import ExprStmt

        call = self._agent_call()
        stmt = ExprStmt(expr=call, span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 1

    def test_raise_traverses_exc_expr(self) -> None:
        """Raise: the exception expression is traversed."""
        from agm.agl.syntax.nodes import Raise

        call = self._agent_call()
        stmt = Raise(exc=call, span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 1

    def test_do_until_traverses_body_and_condition(self) -> None:
        """DoUntil: both the loop body and the loop condition are traversed."""
        from agm.agl.syntax.nodes import DoUntil, ExprStmt

        call_body = self._agent_call()
        call_cond = self._agent_call()
        body_stmt = ExprStmt(expr=call_body, span=self._sp(), node_id=self._nid())
        stmt = DoUntil(
            limit=None, body=(body_stmt,), condition=call_cond,
            span=self._sp(), node_id=self._nid(),
        )
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 2

    def test_if_stmt_traverses_branch_bodies(self) -> None:
        """IfStmt: all branch bodies are traversed."""
        from agm.agl.syntax.nodes import ExprStmt, IfBranch, IfStmt

        call = self._agent_call()
        body_stmt = ExprStmt(expr=call, span=self._sp(), node_id=self._nid())
        branch = IfBranch(
            cond=self._intlit(), body=(body_stmt,), span=self._sp(), node_id=self._nid()
        )
        stmt = IfStmt(branches=(branch,), span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 1

    def test_case_stmt_traverses_subject_and_branch_bodies(self) -> None:
        """CaseStmt: the subject expression and all branch bodies are traversed."""
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, ExprStmt

        call_subj = self._agent_call()
        call_body = self._agent_call()
        body_stmt = ExprStmt(expr=call_body, span=self._sp(), node_id=self._nid())
        branch = CaseStmtBranch(
            pattern=self._wildcard_pattern(), body=(body_stmt,),
            span=self._sp(), node_id=self._nid(),
        )
        stmt = CaseStmt(subject=call_subj, branches=(branch,), span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 2

    def test_try_catch_traverses_body_and_handlers(self) -> None:
        """TryCatch: both try body and all catch-clause bodies are traversed."""
        from agm.agl.syntax.nodes import CatchClause, ExprStmt, TryCatch

        call_try = self._agent_call()
        call_catch = self._agent_call()
        try_stmt = ExprStmt(expr=call_try, span=self._sp(), node_id=self._nid())
        catch_stmt = ExprStmt(expr=call_catch, span=self._sp(), node_id=self._nid())
        clause = CatchClause(
            exc_type=None, binding=None, body=(catch_stmt,),
            span=self._sp(), node_id=self._nid(),
        )
        stmt = TryCatch(body=(try_stmt,), handlers=(clause,), span=self._sp(), node_id=self._nid())
        prog = self._program(stmt)
        assert len(self._collect(prog)) == 2

    def test_nested_program_in_stmts_is_traversed(self) -> None:
        """Program appearing as a statement (nested) is recursed into."""
        from agm.agl.syntax.nodes import ExprStmt, Program

        call = self._agent_call()
        inner_stmt = ExprStmt(expr=call, span=self._sp(), node_id=self._nid())
        inner_prog = Program(body=(inner_stmt,), span=self._sp(), node_id=self._nid())
        outer_prog = Program(body=(inner_prog,), span=self._sp(), node_id=self._nid())
        assert len(self._collect(outer_prog)) == 1

    def test_unrecognized_stmt_type_is_silently_skipped(self) -> None:
        """Statements of unknown types (e.g. PassStmt) are silently ignored."""
        from agm.agl.syntax.nodes import PassStmt, Program

        pass_stmt = PassStmt(span=self._sp(), node_id=self._nid())
        prog = Program(body=(pass_stmt,), span=self._sp(), node_id=self._nid())
        assert self._collect(prog) == []

    def test_call_inventory_default_policy(self) -> None:
        """Agent call with no parse_policy shows 'default' in inventory."""
        from agm.agl.runtime.runtime import WorkflowRuntime as RT

        rt = RT(default_agent=lambda req: req.prompt)
        src = 'let x = prompt "Q"\n'
        result = rt.run(src, check_only=True)
        assert result.ok is True
        assert len(result.call_sites) == 1
        assert result.call_sites[0].parse_policy == "default"

    def test_abort_policy_produces_abort_string_in_inventory(self) -> None:
        """AbortPolicy produces parse_policy='abort' in the call-site inventory."""
        from agm.agl.runtime.runtime import CallSiteInfo, _build_call_inventory
        from agm.agl.syntax.nodes import (
            AbortPolicy,
            AgentCall,
            CallOptions,
            LetDecl,
            Program,
            Template,
            TextSegment,
        )
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        sp = self._sp()

        def nid() -> int:
            return self._nid()

        call_id = nid()
        abort = AbortPolicy(span=sp, node_id=nid())
        opts = CallOptions(
            format=None, strict_json=None, parse_policy=abort, span=sp, node_id=nid()
        )
        seg = TextSegment(text="Q", span=sp, node_id=nid())
        tmpl = Template(segments=(seg,), span=sp, node_id=nid())
        call = AgentCall(agent="prompt", options=opts, template=tmpl, span=sp, node_id=call_id)
        let = LetDecl(name="x", type_ann=None, value=call, span=sp, node_id=nid())
        prog = Program(body=(let,), span=sp, node_id=nid())

        spec = OutputContractSpec(target_type=TextType(), codec_name="text", strict_json=None)

        class FakeChecked:
            contract_specs = {call_id: spec}

        inventory = _build_call_inventory(FakeChecked(), {}, prog)
        assert len(inventory) == 1
        assert isinstance(inventory[0], CallSiteInfo)
        assert inventory[0].parse_policy == "abort"

    def test_missing_spec_skips_call_site(self) -> None:
        """An AgentCall whose node_id is absent from contract_specs is skipped."""
        from agm.agl.runtime.runtime import _build_call_inventory
        from agm.agl.syntax.nodes import (
            AgentCall,
            CallOptions,
            LetDecl,
            Program,
            Template,
            TextSegment,
        )

        sp = self._sp()

        def nid() -> int:
            return self._nid()

        opts = CallOptions(
            format=None, strict_json=None, parse_policy=None, span=sp, node_id=nid()
        )
        seg = TextSegment(text="Q", span=sp, node_id=nid())
        tmpl = Template(segments=(seg,), span=sp, node_id=nid())
        call = AgentCall(agent="prompt", options=opts, template=tmpl, span=sp, node_id=nid())
        let = LetDecl(name="x", type_ann=None, value=call, span=sp, node_id=nid())
        prog = Program(body=(let,), span=sp, node_id=nid())

        class FakeChecked:
            contract_specs: dict[int, object] = {}  # no spec for call

        inventory = _build_call_inventory(FakeChecked(), {}, prog)
        assert inventory == []

    def test_retry_policy_produces_retry_string_in_inventory(self) -> None:
        """RetryPolicy produces parse_policy='retry[N]' in the call-site inventory."""
        from agm.agl.runtime.runtime import CallSiteInfo, _build_call_inventory
        from agm.agl.syntax.nodes import (
            AgentCall,
            CallOptions,
            LetDecl,
            Program,
            RetryPolicy,
            Template,
            TextSegment,
        )
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        sp = self._sp()

        def nid() -> int:
            return self._nid()

        call_id = nid()
        retry = RetryPolicy(extra=5, span=sp, node_id=nid())
        opts = CallOptions(
            format=None, strict_json=None, parse_policy=retry, span=sp, node_id=nid()
        )
        seg = TextSegment(text="Q", span=sp, node_id=nid())
        tmpl = Template(segments=(seg,), span=sp, node_id=nid())
        call = AgentCall(agent="prompt", options=opts, template=tmpl, span=sp, node_id=call_id)
        let = LetDecl(name="x", type_ann=None, value=call, span=sp, node_id=nid())
        prog = Program(body=(let,), span=sp, node_id=nid())

        spec = OutputContractSpec(target_type=TextType(), codec_name="text", strict_json=None)

        class FakeChecked:
            contract_specs = {call_id: spec}

        inventory = _build_call_inventory(FakeChecked(), {}, prog)
        assert len(inventory) == 1
        assert isinstance(inventory[0], CallSiteInfo)
        assert inventory[0].parse_policy == "retry[5]"
