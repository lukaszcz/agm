"""Tests for WorkflowRuntime — M0 shell behaviors preserved, M1 additions.

Covers:
- WorkflowRuntime constructor with default kwargs
- register_agent: duplicate rejection, reserved-name rejection (ask, exec)
- run: full pipeline now active; valid programs succeed; static errors fail
- Diagnostic has .message (str) and .line (int)
- RunResult has .ok (bool), .diagnostics (list), .error (None for pre-exec failures)
- AglError and SourceSpan
- Token constants
- M1 additions: agent registration/fallback, capability derivation, input validation,
  text-codec behavior, AgentCallError, empty-response valid case
"""

from __future__ import annotations

import os
import pathlib

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
        # The source declares the registered agent so the source↔host contract
        # holds (M4); a valid program then returns ok=True.
        result = rt.run("agent reviewer\nlet x = 1")
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

    def test_register_reserved_name_ask_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="ask"):
            rt.register_agent("ask", my_agent)

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
    """Default-agent backing behavior for capability checking."""

    def test_no_default_agent_ask_call_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default_agent
        result = rt.run('let x = ask "hi"')
        assert result.ok is False
        assert result.error is None  # static, not runtime

    def test_with_default_agent_ask_call_succeeds(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('let x = ask "hi"')
        assert result.ok is True

    def test_named_agent_registered_accepted(self) -> None:
        rt = WorkflowRuntime()
        rt.register_agent("impl", lambda req: "output")
        result = rt.run('agent impl\nlet x = impl "do it"')
        assert result.ok is True

    def test_undeclared_named_agent_is_static_error(self) -> None:
        rt = WorkflowRuntime()
        # An undeclared named agent is a static scope binding error: it is
        # rejected before execution regardless of host backing.
        result = rt.run('let x = mysterious_agent "hi"')
        assert result.ok is False
        assert result.error is None

    def test_default_agent_backs_declared_name(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # A default_agent backs any declared name without a dedicated registration.
        result = rt.run('agent any_agent_name\nlet x = any_agent_name "hi"')
        assert result.ok is True

    def test_declared_but_uncalled_agent_surfaces_warning(self) -> None:
        # A default agent backs the declared (but uncalled) agent so the
        # source↔host contract holds (decision 11): a declared+backed agent
        # that is never called is a non-fatal scope WARNING, surfaced on
        # result.warnings without affecting result.ok.
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('agent unused_helper\nprint "hi"')
        assert result.ok is True
        joined = " ".join(d.message for d in result.warnings)
        assert "unused_helper" in joined


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
        rt.run("input x\nlet y = ask \"Hi\"", inputs={})
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

    def test_missing_input_reports_declaration_line(self) -> None:
        """F3: the missing-input diagnostic carries the declaration's line."""
        rt = WorkflowRuntime()
        # ``input spec`` is on line 3; the diagnostic must report line 3, not 1.
        src = "let a = 1\nlet b = 2\ninput spec\nprint spec"
        result = rt.run(src, inputs={})
        assert result.ok is False
        missing = [d for d in result.diagnostics if "spec" in d.message.lower()]
        assert missing, result.diagnostics
        assert missing[0].line == 3

    def test_invalid_typed_input_reports_declaration_line(self) -> None:
        """F3 parity: the type-invalid diagnostic already reports the line."""
        rt = WorkflowRuntime()
        src = "let a = 1\nlet b = 2\ninput n: int\nprint n"
        result = rt.run(src, inputs={"n": "five"})
        assert result.ok is False
        bad = [d for d in result.diagnostics if "n" in d.message.lower()]
        assert bad, result.diagnostics
        assert bad[0].line == 3


class TestEmptyResponse:
    """Exit 0 with empty stdout is a valid empty response (plan §9.5)."""

    def test_empty_string_response_is_valid_text(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "")
        result = rt.run('let x = ask "Say nothing."')
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
        rt.run('let x = ask "Hello world"')
        assert received[0].prompt == "Hello world"

    def test_request_agent_name_for_default(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('let x = ask "Hi"')
        assert received[0].agent == "ask"

    def test_request_agent_name_for_named(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('agent reviewer\nlet x = reviewer "Review this"')
        assert received[0].agent == "reviewer"


class TestUncaughtAgentCallErrorSpan:
    """F2: an uncaught AgentCallError carries the agent-call site's location.

    ``AgentRegistry.dispatch`` raises ``AglRaise`` without a span; the
    interpreter must attach the agent-call node's span so the exit-2 error
    reports ``at line N`` (design §12.6).
    """

    def _failing_runtime(self) -> WorkflowRuntime:
        from agm.agl.runtime.agents import AgentCallHostError

        def failing_agent(req: AgentRequest) -> str:
            raise AgentCallHostError(
                cause="spawn_failure",
                exit_code=None,
                stderr_tail="boom",
                elapsed=0.0,
            )

        return WorkflowRuntime(default_agent=failing_agent)

    def test_dispatch_preserves_existing_span(self) -> None:
        """A span the raise site already supplied is never overwritten."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        existing = SourceSpan(
            start_line=99,
            start_col=1,
            end_line=99,
            end_col=2,
            start_offset=0,
            end_offset=1,
        )

        def agent(req: AgentRequest) -> str:
            exc_val = ExceptionValue(
                type_name="CustomError",
                fields={"message": TextValue("boom")},
            )
            raise AglRaise(exc_val, span=existing)

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let a = 1\nlet x = ask "hi"')
        assert result.ok is False
        assert result.error is not None
        # The agent's own span (line 99) is kept, not replaced by the call site.
        assert result.error.line == 99

    def test_uncaught_agent_call_error_reports_call_line(self) -> None:
        rt = self._failing_runtime()
        # The ``ask`` call is on line 2.
        result = rt.run('let a = 1\nlet x = ask "hi"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentCallError"
        assert result.error.line == 2

    def test_uncaught_agent_call_error_surfaces_at_line_in_message(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: "pathlib.Path",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End to end: ``agm exec`` prints ``at line N`` to stderr (exit 2)."""
        import agm.commands.exec as exec_mod
        from agm.agl.runtime.agents import AgentCallHostError
        from agm.commands.args import ExecArgs
        from agm.commands.exec import run as exec_run

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let a = 1\nlet x = ask "hi"\n')

        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause="spawn_failure", exit_code=None, stderr_tail="boom", elapsed=0.0
            )

        monkeypatch.setattr(
            exec_mod, "runner_backed_agent_factory", lambda **_: failing_agent
        )
        args = ExecArgs(
            file=str(agl_file),
            inputs=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_run(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "at line 2" in err


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
        # ok may be True even when warnings are present: warnings live on their
        # own channel and never appear in ``diagnostics`` (errors-only).
        warning = Diagnostic(message="exhaustiveness", line=4, severity="warning")
        result = RunResult(ok=True, diagnostics=[], error=None, warnings=[warning])
        assert result.ok is True
        assert result.diagnostics == []
        assert result.warnings == [warning]
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
        assert "ask" not in KEYWORDS
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
    """F1a/F1b: an ``ask`` call needs a default (or fallback) agent."""

    def test_ask_without_default_agent_is_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default agent configured
        result = rt.run('let x = ask "hi"')
        assert result.ok is False
        assert result.error is None  # static (pre-execution), not an AgL exception
        assert any("default agent" in d.message.lower() for d in result.diagnostics)

    def test_ask_with_default_agent_runs(self) -> None:
        def agent(request: object) -> str:
            return "answer"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = ask "hi"')
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
        result = rt.run('let x = ask "hi"', check_only=True)
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
        from agm.agl.runtime.runtime import exception_value_to_run_error

        exc = ExceptionValue(
            type_name="ValidationError",
            fields={
                "message": TextValue("bad"),
                "amount": DecimalValue(decimal.Decimal("0.1")),
            },
        )
        err = exception_value_to_run_error(exc)
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
                call_sites=checked.call_sites,
                warnings=(*checked.warnings, warning),
                type_env=checked.type_env,
            )

        monkeypatch.setattr(tc_mod, "check", check_with_warning)

        rt = WorkflowRuntime()
        result = rt.run("input msg\nprint msg", inputs={})
        assert result.ok is False
        # The warning is threaded onto its own channel even on a failure path.
        warning_messages = [d.message for d in result.warnings]
        assert any("a checker warning" in m for m in warning_messages)
        # The missing-input error lands in diagnostics (errors only).
        error_messages = [d.message for d in result.diagnostics]
        assert any("msg" in m for m in error_messages)
        # Channels stay separate: no warning leaks into diagnostics.
        assert all(d.severity == "error" for d in result.diagnostics)


class TestAgentRegistryDispatch:
    """F17: dispatch resolves named agents, ask, and the default fallback."""

    def test_dispatch_named_agent(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def named(req: AgentRequest) -> str:
            return f"named:{req.prompt}"

        registry = AgentRegistry(named={"reviewer": named}, default_agent=None)
        resp = registry.dispatch("reviewer", AgentRequest(agent="reviewer", prompt="hi"))
        assert resp.content == "named:hi"

    def test_dispatch_ask_and_unknown_fall_back_to_default(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def default(req: AgentRequest) -> str:
            return f"default:{req.agent}"

        registry = AgentRegistry(named={}, default_agent=default)
        # Both ``ask`` and an unregistered named agent route to the default.
        assert registry.dispatch("ask", AgentRequest(agent="ask", prompt="q")).content == (
            "default:ask"
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


# ---------------------------------------------------------------------------
# CARRY-IN 1 — capabilities built from registrations (M3b)
# ---------------------------------------------------------------------------


class TestCapabilitiesBuiltFromRegistrations:
    """CARRY-IN 1: WorkflowRuntime.run builds HostCapabilities from codec/renderer registries."""

    def test_default_runtime_has_text_and_json_codecs(self) -> None:
        """Built-in text + json codecs are always present."""
        from agm.agl.runtime.codec import JsonCodec, TextCodec

        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # A json-typed call passes typecheck → json codec is registered.
        tc, jc = TextCodec(), JsonCodec()
        assert tc.name == "text"
        assert jc.name == "json"
        result = rt.run('let x = ask "hi"')
        assert result.ok is True

    def test_default_runtime_has_builtin_renderer_names(self) -> None:
        """Built-in renderers (default, raw, json, bullets) are always present."""
        from agm.agl.runtime.render import RENDERER_NAMES
        assert frozenset({"default", "raw", "json", "bullets"}) <= RENDERER_NAMES

    def test_register_codec_before_run_extends_capabilities(self) -> None:
        """A custom codec registered before run() makes its kinds available to typecheck."""
        from agm.agl.eval.values import TextValue as TV
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.typecheck.types import TextType, Type

        class FooCodec:
            @property
            def name(self) -> str:
                return "foo"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(self, type_ref: Type) -> OutputContract:
                return OutputContract(
                    target_type=type_ref,
                    codec=TextCodec(),
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                return ParseResult.success(TV(raw))

        rt = WorkflowRuntime()
        rt.register_codec(FooCodec())
        # The program just needs to run without capability errors.
        result = rt.run("let x = 1")
        assert result.ok is True

    def test_registered_renderer_makes_interpolation_typecheck(self) -> None:
        """``${x as fancy}`` typechecks ONLY when ``fancy`` is registered (F4)."""
        src = 'input x\nlet y = ask "see ${x as fancy}"'

        # Without registration: the renderer is unknown → static type error.
        rt_unreg = WorkflowRuntime(default_agent=lambda req: "ok")
        unreg = rt_unreg.run(src, inputs={"x": "hi"})
        assert unreg.ok is False
        assert any("fancy" in d.message for d in unreg.diagnostics)

        # With registration: the same program now passes static checking.
        rt_reg = WorkflowRuntime(default_agent=lambda req: "ok")
        rt_reg.register_renderer("fancy", lambda val, name: str(val))
        reg = rt_reg.run(src, inputs={"x": "hi"})
        assert reg.ok is True

    def test_unregistered_renderer_is_a_static_error(self) -> None:
        """An ``as <name>`` for an unregistered renderer is rejected (F4)."""
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run(
            'input x\nlet y = ask "${x as nope}"', inputs={"x": "hi"}
        )
        assert result.ok is False
        assert any("nope" in d.message for d in result.diagnostics)

    def test_no_duplicated_constant_in_runtime(self) -> None:
        """runtime.py derives renderer names from the render module, not a literal.

        Behavioral check: a renderer name that is a built-in (``json``) is
        accepted in an interpolation without any registration, proving the
        runtime sources the built-in set from ``render`` (F4).
        """
        from agm.agl.runtime.render import RENDERER_NAMES

        assert isinstance(RENDERER_NAMES, frozenset)
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # ``json`` is a built-in renderer → accepted with no registration.
        result = rt.run('input x\nlet y = ask "${x as json}"', inputs={"x": "hi"})
        assert result.ok is True


# ---------------------------------------------------------------------------
# Coverage: render.py — render_for_prompt / render_for_console / helpers
# ---------------------------------------------------------------------------


class TestRenderForPrompt:
    """Direct unit tests for render_for_prompt and render_for_console."""

    def test_text_default_renderer_has_boundary(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(TextValue("hello"), renderer_name=None, var_name="x")
        assert "hello" in out
        assert 'name="x"' in out
        assert 'type="text"' in out

    def test_text_default_renderer_no_varname(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(TextValue("hi"), renderer_name=None, var_name=None)
        assert "hi" in out
        assert 'name="value"' in out

    def test_int_default_renderer_scalar(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(IntValue(42), renderer_name=None, var_name="n")
        assert out == "42"

    def test_decimal_default_renderer_scalar(self) -> None:
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(DecimalValue(Decimal("1.5")), renderer_name=None, var_name="d")
        assert out == "1.5"

    def test_bool_default_renderer_scalar(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import render_for_prompt

        assert render_for_prompt(BoolValue(True), renderer_name=None, var_name="b") == "true"
        assert render_for_prompt(BoolValue(False), renderer_name=None, var_name="b") == "false"

    def test_json_default_renderer_structured(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(JsonValue({"k": 1}), renderer_name=None, var_name="j")
        assert "k" in out
        assert 'type="json"' in out

    def test_list_default_renderer_structured(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(
            ListValue(elements=(IntValue(1), IntValue(2))),
            renderer_name=None,
            var_name="xs",
        )
        assert 'type="list"' in out

    def test_dict_default_renderer_structured(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(
            DictValue(entries={"a": IntValue(1)}),
            renderer_name=None,
            var_name="d",
        )
        assert 'type="dict"' in out

    def test_record_default_renderer_uses_type_name(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import render_for_prompt

        rv = RecordValue(type_name="Issue", fields={"n": IntValue(3)})
        out = render_for_prompt(rv, renderer_name=None, var_name="r")
        assert 'type="Issue"' in out

    def test_enum_default_renderer_uses_type_name(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_for_prompt

        ev = EnumValue(type_name="Status", variant="Done", fields={})
        out = render_for_prompt(ev, renderer_name=None, var_name="s")
        assert 'type="Status"' in out

    def test_raw_renderer(self) -> None:
        from agm.agl.eval.values import IntValue, TextValue
        from agm.agl.runtime.render import render_for_prompt

        assert render_for_prompt(TextValue("hi"), renderer_name="raw", var_name="x") == "hi"
        assert render_for_prompt(IntValue(7), renderer_name="raw", var_name="n") == "7"

    def test_json_renderer(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(IntValue(5), renderer_name="json", var_name="n")
        assert "5" in out

    def test_bullets_renderer_list(self) -> None:
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(
            ListValue(elements=(TextValue("a"), TextValue("b"))),
            renderer_name="bullets",
            var_name="xs",
        )
        assert "- a" in out
        assert "- b" in out

    def test_bullets_renderer_non_list(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        # Non-list falls through to pretty JSON.
        out = render_for_prompt(IntValue(3), renderer_name="bullets", var_name="n")
        assert "3" in out

    def test_unknown_renderer_raises_internal_error(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        # An unknown renderer name is a checker-invariant violation, not a silent
        # fallback (F2, M3b): render_for_prompt fails loudly so a broken
        # renderers table cannot masquerade as default output.
        with pytest.raises(AssertionError, match="notarenderer"):
            render_for_prompt(
                TextValue("x"),
                renderer_name="notarenderer",
                var_name="v",
                renderers={},
            )

    def test_render_for_console_text(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_console

        assert render_for_console(TextValue("hello")) == "hello"

    def test_render_for_console_int(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_console

        assert render_for_console(IntValue(7)) == "7"

    def test_render_for_console_decimal(self) -> None:
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import render_for_console

        assert render_for_console(DecimalValue(Decimal("1.5"))) == "1.5"

    def test_render_for_console_bool(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import render_for_console

        assert render_for_console(BoolValue(True)) == "true"
        assert render_for_console(BoolValue(False)) == "false"

    def test_render_for_console_json(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_console

        out = render_for_console(JsonValue({"k": 1}))
        assert "k" in out

    def test_render_for_console_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_console

        out = render_for_console(ListValue(elements=(IntValue(1),)))
        assert "1" in out

    def test_render_for_console_record(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import render_for_console

        out = render_for_console(RecordValue(type_name="R", fields={"x": IntValue(3)}))
        assert "x" in out

    def test_render_for_console_enum(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_for_console

        out = render_for_console(EnumValue(type_name="E", variant="A", fields={}))
        assert "A" in out

    def test_render_for_console_dict(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import render_for_console

        out = render_for_console(DictValue(entries={"k": TextValue("v")}))
        assert "k" in out

    def test_scalar_text_json_value(self) -> None:
        """_scalar_text(JsonValue) delegates to dumps_exact (line 67-68)."""
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import _scalar_text

        out = _scalar_text(JsonValue({"a": 1}))
        assert "a" in out

    def test_scalar_text_list_value_falls_back_to_pretty_json(self) -> None:
        """_scalar_text with a ListValue falls back to _pretty_json (line 69)."""
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import _scalar_text

        out = _scalar_text(ListValue(elements=(IntValue(1),)))
        assert "1" in out

    def test_type_kind_str_scalar_types(self) -> None:
        """_type_kind_str returns the correct kind for each scalar type."""
        from decimal import Decimal

        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            ExceptionValue,
            IntValue,
            TextValue,
        )
        from agm.agl.runtime.render import _type_kind_str

        assert _type_kind_str(TextValue("x")) == "text"
        assert _type_kind_str(IntValue(1)) == "int"
        assert _type_kind_str(DecimalValue(Decimal("1.5"))) == "decimal"
        assert _type_kind_str(BoolValue(True)) == "bool"
        assert _type_kind_str(ExceptionValue(type_name="Boom", fields={})) == "Boom"

    def test_exception_default_renderer_fenced_pretty_json(self) -> None:
        """§8.1 / §2.12: ${e} interpolation renders as fenced pretty JSON.

        When an exception value is interpolated via the default renderer, the
        output must be fenced with <dsl-value> boundary tags carrying the
        exception's type name and contain the field values as pretty JSON.
        This pins the ${e} rendering behavior described in §8.1 (whole-value
        interpolation) and §2.12 (exceptions → fenced pretty JSON).
        """
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import render_for_prompt

        exc_val = ExceptionValue(
            type_name="Abort",
            fields={
                "message": TextValue("fatal"),
                "trace_id": TextValue("abc123"),
            },
        )
        out = render_for_prompt(exc_val, renderer_name=None, var_name="e")
        # Must be boundary-marked with the exception's type name.
        assert 'type="Abort"' in out, f"Expected type=Abort in output: {out!r}"
        assert 'name="e"' in out, f"Expected name=e in output: {out!r}"
        # Must contain the field values as JSON (fenced pretty JSON).
        assert "fatal" in out, f"Expected message value in output: {out!r}"
        assert "abc123" in out, f"Expected trace_id value in output: {out!r}"
        # Must be wrapped in dsl-value tags.
        assert "<dsl-value" in out and "</dsl-value>" in out, (
            f"Expected dsl-value tags in output: {out!r}"
        )


# ---------------------------------------------------------------------------
# §2.12: type= attribute finalization — list/dict/<TypeName> convention
# ---------------------------------------------------------------------------


class TestTypeAttributeFinalization:
    """§2.12 finalization: verify type= attribute convention for boundary tags.

    §2.12 says a conforming runtime may choose its boundary format (stable
    and traceable).  Ruling: list/dict/<TypeName> convention.  These tests
    pin the finalized convention after removing the provisional marker.
    """

    def test_list_type_attribute_is_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(
            ListValue(elements=(IntValue(1),)), renderer_name=None, var_name="xs"
        )
        assert 'type="list"' in out, f"Expected type=list: {out!r}"

    def test_dict_type_attribute_is_dict(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import render_for_prompt

        out = render_for_prompt(
            DictValue(entries={"k": TextValue("v")}), renderer_name=None, var_name="d"
        )
        assert 'type="dict"' in out, f"Expected type=dict: {out!r}"

    def test_record_type_attribute_is_type_name(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import render_for_prompt

        rv = RecordValue(type_name="Review", fields={"score": IntValue(5)})
        out = render_for_prompt(rv, renderer_name=None, var_name="r")
        assert 'type="Review"' in out, f"Expected type=Review: {out!r}"

    def test_enum_type_attribute_is_type_name(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_for_prompt

        ev = EnumValue(type_name="Status", variant="Pass", fields={})
        out = render_for_prompt(ev, renderer_name=None, var_name="s")
        assert 'type="Status"' in out, f"Expected type=Status: {out!r}"


# ---------------------------------------------------------------------------
# Coverage: serialize.py — value_to_json_obj and dumps_exact branches
# ---------------------------------------------------------------------------


class TestSerialize:
    """Coverage for serialize.py branches not exercised by higher-level tests."""

    def test_bool_value_serialized(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.serialize import value_to_json_obj

        assert value_to_json_obj(BoolValue(True)) is True
        assert value_to_json_obj(BoolValue(False)) is False

    def test_dict_value_serialized(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(DictValue(entries={"a": IntValue(1)}))
        assert result == {"a": 1}

    def test_record_value_serialized(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(RecordValue(type_name="R", fields={"x": IntValue(5)}))
        assert result == {"x": 5}

    def test_enum_value_serialized(self) -> None:
        from agm.agl.eval.values import EnumValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(
            EnumValue(type_name="E", variant="A", fields={"msg": TextValue("hi")})
        )
        assert result == {"$case": "A", "msg": "hi"}

    def test_enum_nullary_value_serialized(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(EnumValue(type_name="E", variant="Done", fields={}))
        assert result == {"$case": "Done"}

    def test_exception_value_serialized(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(
            ExceptionValue(type_name="Err", fields={"message": TextValue("oops")})
        )
        assert result == {"message": "oops"}

    def test_dumps_exact_bool_true(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact(True) == "true"

    def test_dumps_exact_bool_false(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact(False) == "false"

    def test_dumps_exact_list_empty(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact([]) == "[]"

    def test_dumps_exact_list_no_indent(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        result = dumps_exact([1, 2], indent=None)
        assert "1" in result
        assert "2" in result

    def test_dumps_exact_dict_empty(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact({}) == "{}"

    def test_dumps_exact_dict_no_indent(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        result = dumps_exact({"k": 1}, indent=None)
        assert "k" in result
        assert "1" in result


# ---------------------------------------------------------------------------
# Coverage: agents.py — AgentResponse returned directly (not str)
# ---------------------------------------------------------------------------


class TestAgentResponseDirectReturn:
    """Cover the branch in AgentRegistry.dispatch that returns AgentResponse directly."""

    def test_agent_returns_agent_response_directly(self) -> None:
        from agm.agl.runtime import AgentRequest, AgentResponse
        from agm.agl.runtime.agents import AgentRegistry

        def agent_fn(req: AgentRequest) -> AgentResponse:
            return AgentResponse(content="direct", metadata={"k": "v"})

        registry = AgentRegistry(named={"myagent": agent_fn}, default_agent=None)
        result = registry.dispatch("myagent", AgentRequest(agent="myagent", prompt="q"))
        assert result.content == "direct"
        assert result.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# Coverage: contract.py — ValueError when codec not found
# ---------------------------------------------------------------------------


class TestMaterializeContractMissingCodec:
    """Cover the ValueError branch when the codec is not in the registry."""

    def test_missing_codec_raises_value_error(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        spec = OutputContractSpec(
            target_type=TextType(),
            codec_name="nonexistent_codec",
            strict_json=None,
        )
        with pytest.raises(ValueError, match="nonexistent_codec"):
            materialize_contract(spec, {"text": TextCodec()})


# ---------------------------------------------------------------------------
# Coverage: runtime.py — generic exception handlers and error paths
# ---------------------------------------------------------------------------


class TestRuntimeErrorPaths:
    """Cover the generic exception handler branches in WorkflowRuntime.run."""

    def test_generic_parse_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglSyntaxError) exception in parse step → ok=False."""
        import agm.agl.parser as parser_mod

        def bad_parse(source: str) -> object:
            raise RuntimeError("unexpected parse error")

        monkeypatch.setattr(parser_mod, "parse_program", bad_parse)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("unexpected parse error" in d.message for d in result.diagnostics)

    def test_tab_warning_included_even_on_parse_failure(self) -> None:
        """Tab advisories come from the lexer's single scan, so they survive a
        parse failure: the scan completes (recording the TAB) before the grammar
        rejects the token stream."""
        rt = WorkflowRuntime()
        result = rt.run("\tprint")  # leading TAB, then an incomplete `print`
        assert result.ok is False
        assert result.diagnostics  # genuine parse error surfaced
        tab_warns = [w for w in result.warnings if w.severity == "warning"]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 1

    def test_generic_scope_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglScopeError) exception in scope step → ok=False."""
        import agm.agl.scope as scope_mod

        def bad_resolve(program: object) -> object:
            raise RuntimeError("unexpected scope error")

        monkeypatch.setattr(scope_mod, "resolve", bad_resolve)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("Scope error" in d.message for d in result.diagnostics)

    def test_generic_typecheck_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglTypeError) exception in typecheck step → ok=False."""
        import agm.agl.typecheck as tc_mod

        def bad_check(resolved: object, caps: object) -> object:
            raise RuntimeError("unexpected type error")

        monkeypatch.setattr(tc_mod, "check", bad_check)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("Type error" in d.message for d in result.diagnostics)

    def test_contract_error_returns_not_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract materialization error → ok=False with contract error diagnostic."""
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object) -> object:
            raise ValueError("bad contract")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('let x = ask "hi"')
        assert result.ok is False
        assert any("Contract error" in d.message for d in result.diagnostics)

    def test_uncaught_agl_raise_in_run(self) -> None:
        """AglRaise propagating from the interpreter → RunResult with error."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        def bad_agent(req: object) -> str:
            raise AglRaise(
                ExceptionValue(
                    type_name="Abort",
                    fields={"message": TextValue("stopped"), "trace_id": TextValue("")},
                )
            )

        rt = WorkflowRuntime(default_agent=bad_agent)
        result = rt.run('let x = ask "hi"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"

    def test_text_input_not_str_raises(self) -> None:
        """convert_input: text type with non-str value → ValueError."""
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import TextType

        with pytest.raises(ValueError, match="expected a text value"):
            convert_input("msg", 42, TextType())

    def test_int_input_decimal_integral_widened(self) -> None:
        """convert_input: integral Decimal → IntValue for int type."""
        from decimal import Decimal

        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import IntType

        result = convert_input("n", Decimal("3"), IntType())
        assert result == IntValue(3)

    def test_int_input_non_integral_fails(self) -> None:
        """convert_input: non-integral value → ValueError for int type."""
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="expected an integer"):
            convert_input("n", "1.5", IntType())

    def test_decimal_input_from_int(self) -> None:
        """convert_input: int value → DecimalValue for decimal type."""
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import DecimalType

        result = convert_input("d", 3, DecimalType())
        assert isinstance(result, DecimalValue)
        assert result.value == Decimal(3)

    def test_decimal_input_invalid_type_fails(self) -> None:
        """convert_input: bool value → ValueError for decimal type."""
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import DecimalType

        with pytest.raises(ValueError, match="expected a decimal"):
            convert_input("d", "true", DecimalType())

    def test_bool_input_invalid_type_fails(self) -> None:
        """convert_input: non-bool value → ValueError for bool type."""
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import BoolType

        with pytest.raises(ValueError, match="expected a bool"):
            convert_input("b", "1", BoolType())

    def test_bool_input_true_succeeds(self) -> None:
        """convert_input: bool value → BoolValue for bool type."""
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import BoolType

        result = convert_input("b", True, BoolType())
        assert result == BoolValue(True)

    # --- assertions migrated from TestRuntimeExceptionHandlers (eval tests) ---

    def test_internal_interpreter_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1c: an unexpected (non-AglRaise) interpreter error must propagate.

        A Python-level bug must crash loudly rather than masquerade as a
        user-facing pre-execution diagnostic.
        """
        from agm.agl.eval.interpreter import Interpreter

        def bad_execute(self: Interpreter, root_scope: object) -> None:
            raise RuntimeError("internal crash")

        monkeypatch.setattr(Interpreter, "execute", bad_execute)

        rt = WorkflowRuntime()
        with pytest.raises(RuntimeError, match="internal crash"):
            rt.run("let x = 1")

    def test_exception_value_to_run_error_maps_all_field_kinds(self) -> None:
        """exception_value_to_run_error converts every Value kind to JSON shape.

        This is the pure converter used to surface an uncaught AgL exception
        (e.g. AgentParseError) as a RunError.
        """
        import decimal

        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            DictValue,
            EnumValue,
            ExceptionValue,
            IntValue,
            JsonValue,
            ListValue,
            RecordValue,
            TextValue,
        )
        from agm.agl.runtime.runtime import RunError, exception_value_to_run_error

        exc_val = ExceptionValue(
            type_name="AgentParseError",
            fields={
                "message": TextValue("failed"),
                "trace_id": TextValue(""),
                "raw": TextValue("abc"),
                "agent": TextValue("ask"),
                "attempts": IntValue(1),
                "target_type": TextValue("text"),
                "decimal_val": DecimalValue(decimal.Decimal("1.5")),
                "bool_val": BoolValue(True),
                "json_val": JsonValue({"k": "v"}),
                "list_val": ListValue(elements=(IntValue(1),)),
                "dict_val": DictValue(entries={"x": IntValue(2)}),
                "rec_val": RecordValue(type_name="R", fields={"f": TextValue("v")}),
                "enum_val": EnumValue(type_name="E", variant="V", fields={}),
                "exc_val": ExceptionValue(type_name="Inner", fields={}),
                "none_val": JsonValue(None),
            },
        )
        error = exception_value_to_run_error(exc_val)
        assert isinstance(error, RunError)
        assert error.type_name == "AgentParseError"
        assert error.fields["message"] == "failed"
        # F3/F9: Decimal is preserved exactly (not converted to float).
        assert error.fields["decimal_val"] == decimal.Decimal("1.5")
        assert isinstance(error.fields["decimal_val"], decimal.Decimal)
        assert error.fields["bool_val"] is True
        assert error.fields["json_val"] == {"k": "v"}
        assert error.fields["list_val"] == [1]
        assert error.fields["dict_val"] == {"x": 2}
        assert error.fields["rec_val"] == {"f": "v"}
        assert error.fields["enum_val"] == {"$case": "V"}
        assert isinstance(error.fields["exc_val"], dict)

    def test_convert_input_json_type_accepts_any(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import JsonType

        result = convert_input("meta", [1, 2, 3], JsonType())
        assert result == JsonValue([1, 2, 3])

    def test_convert_input_list_type_parsed_via_json_codec(self) -> None:
        # M2: list/dict/record/enum inputs are now accepted via the JsonCodec.
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import ListType, TextType

        result = convert_input("xs", '["a", "b"]', ListType(elem=TextType()))
        assert isinstance(result, ListValue)
        assert result.elements == (TextValue("a"), TextValue("b"))

    def test_agl_raise_from_interpreter_becomes_run_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AglRaise from the interpreter → RunResult.error (not None)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.values import ExceptionValue, TextValue

        def bad_execute(self: Interpreter, root_scope: object) -> None:
            exc_val = ExceptionValue(
                type_name="Abort",
                fields={"message": TextValue("fatal"), "trace_id": TextValue("")},
            )
            raise AglRaise(exc_val)

        monkeypatch.setattr(Interpreter, "execute", bad_execute)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert result.error.fields.get("message") == "fatal"

    # --- Task 3: structured Decimal inputs and non-JSON-shaped rejection ---

    def test_list_decimal_input_validates_exactly(self) -> None:
        """Task 3: a list[decimal] input with native Decimal values must bind
        correctly without the old default=str corruption.

        Before the fix, Decimal("1.5") was serialized as the JSON string "1.5"
        (quoted), which failed schema validation.
        """
        import decimal as _decimal

        from agm.agl.eval.values import DecimalValue, ListValue
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import DecimalType, ListType

        result = convert_input(
            "xs", [_decimal.Decimal("1.5"), _decimal.Decimal("2.75")], ListType(elem=DecimalType())
        )
        assert isinstance(result, ListValue)
        assert result.elements == (
            DecimalValue(_decimal.Decimal("1.5")),
            DecimalValue(_decimal.Decimal("2.75")),
        )

    def test_non_json_shaped_object_yields_clean_diagnostic(self) -> None:
        """Task 3: a non-JSON-shaped object (e.g. a set) must yield a clean
        input-validation error naming the input, not a stringified value or
        traceback.
        """
        from agm.agl.runtime.runtime import convert_input
        from agm.agl.typecheck.types import ListType, TextType

        with pytest.raises(ValueError, match="xs") as exc_info:
            convert_input("xs", {1, 2, 3}, ListType(elem=TextType()))
        # The error message must name the input and mention the type, not
        # contain a raw repr of the set or a json.dumps traceback.
        msg = str(exc_info.value)
        assert "set" in msg  # type name named

    def test_decimal_native_in_list_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Task 3 e2e: input xs: list[decimal] with Decimal values binds and prints."""
        import decimal as _decimal

        result = WorkflowRuntime().run(
            "input xs: list[decimal]\nprint xs\n",
            inputs={"xs": [_decimal.Decimal("1.5"), _decimal.Decimal("2.25")]},
        )
        assert result.ok is True
        out = capsys.readouterr().out
        assert "1.5" in out
        assert "2.25" in out

    def test_is_json_shaped_dict_with_non_str_key_is_false(self) -> None:
        """_is_json_shaped: a dict with non-str keys is not JSON-shaped (covers
        the dict branch of _is_json_shaped, line 790).
        """
        from agm.agl.runtime.runtime import _is_json_shaped

        # Dict with non-str key.
        assert _is_json_shaped({1: "a"}) is False
        # Dict with str keys and JSON-shaped values.
        assert _is_json_shaped({"k": 1}) is True


class TestRegisteredRendererInvoked:
    """F1 (M3b): a host-registered renderer is actually invoked at eval time.

    Before F1 the registered renderer was never threaded into the interpreter,
    so ``${x as myrender}`` silently produced the *default* boundary-marked
    rendering.  These end-to-end tests run a program through ``run()`` with a
    stub agent and assert the agent's received prompt contains the custom
    renderer's distinctive output.
    """

    def test_custom_renderer_output_reaches_agent_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        def my_render(value: object, name: str | None) -> str:
            # Distinctive marker the default renderer would never emit.
            return "[[CUSTOM-RENDER]]"

        rt = WorkflowRuntime(default_agent=agent)
        rt.register_renderer("myrender", my_render)
        result = rt.run(
            'input x\nlet y = ask "see: ${x as myrender}"',
            inputs={"x": "ignored-by-custom-renderer"},
        )
        assert result.ok is True
        assert received, "agent should have been called"
        prompt = received[0].prompt
        assert "[[CUSTOM-RENDER]]" in prompt
        # The default text boundary marker must NOT appear: the custom renderer
        # fully replaced the default rendering.
        assert "<dsl-value" not in prompt

    def test_custom_renderer_receives_value_and_name(self) -> None:
        received: list[AgentRequest] = []
        seen: list[tuple[object, str | None]] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        def my_render(value: object, name: str | None) -> str:
            seen.append((value, name))
            from agm.agl.eval.values import TextValue

            assert isinstance(value, TextValue)
            return f"<{value.value}|{name}>"

        rt = WorkflowRuntime(default_agent=agent)
        rt.register_renderer("tag", my_render)
        result = rt.run(
            'input x\nlet y = ask "${x as tag}"', inputs={"x": "payload"}
        )
        assert result.ok is True
        assert seen and seen[0][1] == "x"
        assert "<payload|x>" in received[0].prompt

    def test_builtin_renderer_still_used_when_no_override(self) -> None:
        """A program with no ``as`` override still gets the default rendering."""
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.register_renderer("unused", lambda v, n: "NOPE")
        result = rt.run(
            'input x\nlet y = ask "${x}"', inputs={"x": "hello"}
        )
        assert result.ok is True
        # Default text rendering is boundary-marked; the custom renderer is NOT
        # applied to a plain ``${x}``.
        assert "<dsl-value" in received[0].prompt
        assert "NOPE" not in received[0].prompt


class TestMaxIterationsExceededSchema:
    """F2: ``MaxIterationsExceeded`` carries the full §8.1 field schema.

    The interpreter populates ``condition`` (the until-expression's exact source
    text, recovered via span offsets into the threaded source), the final
    ``last_condition_value``, and a ``metadata`` json placeholder — alongside the
    pre-existing ``limit``.  Exercised end-to-end by catching the exception in an
    AgL program and printing each field.
    """

    _PROGRAM = (
        "var n = 0\n"
        "try\n"
        "  do[2]\n"
        "    set n = n + 1\n"
        "  until n > 10\n"
        "catch MaxIterationsExceeded as e =>\n"
        "  print e.limit\n"
        "  print e.condition\n"
        "  print e.last_condition_value\n"
    )

    def test_fields_surface_through_real_source(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run(self._PROGRAM)
        # The exception is caught, so the run completes successfully.
        assert result.ok is True
        assert result.error is None
        lines = capsys.readouterr().out.splitlines()
        # limit, condition (exact until-expression source), last_condition_value.
        assert lines == ["2", "n > 10", "false"]

    def test_metadata_field_is_accessible(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ``metadata`` is a json placeholder (null until the M4 trace store), but
        # it is part of the schema and must be readable as a field.
        rt = WorkflowRuntime()
        program = (
            "try\n"
            "  do[1] pass until false\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.metadata\n"
        )
        result = rt.run(program)
        assert result.ok is True
        assert capsys.readouterr().out.strip() == "null"

    def test_condition_reflects_each_distinct_until_expression(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A different until-expression must yield a different ``condition`` slice,
        # proving the source text is recovered per-node rather than hard-coded.
        rt = WorkflowRuntime()
        program = (
            "let done = false\n"
            "try\n"
            "  do[1] pass until done\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.condition\n"
            "  print e.last_condition_value\n"
        )
        result = rt.run(program)
        assert result.ok is True
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["done", "false"]


class TestExhaustivenessWarningSurfaces:
    """F1: a non-exhaustive enum ``case`` warns without failing the run.

    The exhaustiveness diagnostic is a warning, so ``ok`` stays ``True`` and the
    warning is visible on ``result.warnings`` (never in ``result.diagnostics``)
    while the program executes.
    """

    def test_warning_surfaces_and_run_succeeds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        program = (
            "enum R\n"
            "  | Pass\n"
            "  | Fail\n"
            "let r: R = Pass\n"
            "case r of\n"
            '  | Pass => print "ok"\n'
        )
        result = rt.run(program)
        # Warning, not error: the run still succeeds.
        assert result.ok is True
        assert result.error is None
        # Successful runs carry no error diagnostics; the warning is separate.
        assert result.diagnostics == []
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"
        assert "Fail" in result.warnings[0].message
        # The matched branch executed.
        assert capsys.readouterr().out == "ok\n"


class TestShellExecTimeoutProperty:
    """M4: shell_exec_timeout is a readable constructor parameter."""

    def test_default_shell_exec_timeout_is_none(self) -> None:
        rt = WorkflowRuntime()
        assert rt.shell_exec_timeout is None

    def test_shell_exec_timeout_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(shell_exec_timeout=30.0)
        assert rt.shell_exec_timeout == 30.0


# Permission-based tests: chmod 0o444 has no effect for root, who can write
# regardless.  Skip there rather than assert a false negative.
_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


class TestTraceWriteFailureIsBestEffort:
    """F2b: a mid-run trace write failure must not corrupt program semantics."""

    def test_emit_failure_warns_once_and_disables_store(
        self,
        tmp_path: "object",
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing ``append_jsonl`` is caught: one stderr warning is emitted,
        the store disables itself, and no further writes are attempted."""
        from pathlib import Path

        from agm.agl.runtime.trace import TraceStore

        store = TraceStore(path=Path(str(tmp_path)) / "trace.log")
        calls = {"n": 0}

        def failing_append(_path: object, _record: object) -> None:
            calls["n"] += 1
            raise OSError("disk gone")

        import agm.agl.runtime.trace as trace_mod

        monkeypatch.setattr(trace_mod, "append_jsonl", failing_append)
        store.run_start()  # first emit → fails, warns, disables
        store.print_stmt(rendered="hi")  # disabled → no further attempt
        store.run_end(ok=True)

        # Only the first emit attempted a write; the rest short-circuit.
        assert calls["n"] == 1
        err = capsys.readouterr().err
        assert err.count("trace logging disabled") == 1
        assert "disk gone" in err

    @_skip_if_root
    def test_run_completes_when_trace_becomes_unwritable_midrun(
        self, tmp_path: "object", capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real chmod-444 mid-run leaves the program semantics intact: every
        statement still runs and the RunResult is a normal success."""
        from pathlib import Path

        log_file = Path(str(tmp_path)) / "trace.log"
        rt = WorkflowRuntime()

        # Pre-create the trace file and make it read-only so the first record
        # write (run_start) fails — the run must still complete normally.
        log_file.write_text("")
        log_file.chmod(0o444)

        program = 'print "a"\nprint "b"\nprint "c"\n'
        result = rt.run(program, log_file=log_file)

        # Program semantics unaffected: clean success, all prints emitted.
        assert result.ok is True
        assert result.error is None
        out = capsys.readouterr()
        assert out.out == "a\nb\nc\n"
        # Exactly one warning about disabled trace logging.
        assert out.err.count("trace logging disabled") == 1


# ---------------------------------------------------------------------------
# TAB character warnings in RunResult
# ---------------------------------------------------------------------------


class TestTabWarningsInRunResult:
    """Tab characters in source produce warning diagnostics in RunResult.warnings."""

    def test_no_tab_no_warning(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert tab_warns == []

    def test_tab_in_valid_source_yields_warning(self) -> None:
        # TAB used as whitespace inside a valid statement on the second line.
        source = "let x = 1\nlet\ty = 2"
        rt = WorkflowRuntime()
        result = rt.run(source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 2

    def test_tab_warning_does_not_affect_ok(self) -> None:
        # A tab warning must not cause ok to become False.
        source = "let\tx = 1"
        rt = WorkflowRuntime()
        result = rt.run(source)
        assert result.ok is True
        assert result.error is None

    def test_multiple_tabs_multiple_warnings(self) -> None:
        source = "let\tx = 1\nlet\ty = 2"
        rt = WorkflowRuntime()
        result = rt.run(source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 2


# ---------------------------------------------------------------------------
# declared_agents() API (M4)
# ---------------------------------------------------------------------------


class TestDeclaredAgentsApi:
    """WorkflowRuntime.declared_agents(): parse + scope only, non-raising."""

    def test_returns_agent_decl_info_with_names_runners_and_positions(self) -> None:
        from agm.agl import AgentDeclInfo

        rt = WorkflowRuntime()
        source = 'agent impl = "claude -p %{PROMPT_FILE}"\nagent reviewer'
        decls = rt.declared_agents(source)
        assert all(isinstance(d, AgentDeclInfo) for d in decls)
        # Sorted deterministically by source line/col.
        assert [d.name for d in decls] == ["impl", "reviewer"]
        impl, reviewer = decls
        assert impl.runner == "claude -p %{PROMPT_FILE}"
        assert reviewer.runner is None
        # Positions come from the declaration span (1-based).
        assert impl.line == 1
        assert impl.col == 1
        assert reviewer.line == 2

    def test_no_declarations_returns_empty(self) -> None:
        rt = WorkflowRuntime()
        assert rt.declared_agents("let x = 1") == ()

    def test_parse_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Syntax garbage: declared_agents stays non-raising and returns ().
        assert rt.declared_agents("@@@@@") == ()

    def test_scope_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Duplicate agent declaration is a scope error → ().
        assert rt.declared_agents("agent dup\nagent dup") == ()

    def test_undeclared_call_scope_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Calling an undeclared agent is a scope error → ().
        assert rt.declared_agents('let x = ghost "hi"') == ()


# ---------------------------------------------------------------------------
# Source↔host reconciliation in run() (M4, plan §8, decisions 1 & 11)
# ---------------------------------------------------------------------------


class TestAgentReconciliation:
    """run() enforces the source↔host agent contract before execution."""

    def test_registered_but_undeclared_is_host_error(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("ghost", agent)
        # 'ghost' is registered but the source never declares it.
        result = rt.run("let x = 1")
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "ghost" in msgs
        assert "registered" in msgs.lower()
        # Nothing executed.
        assert calls == []

    def test_registered_but_undeclared_diagnostic_line_is_one(self) -> None:
        rt = WorkflowRuntime()
        rt.register_agent("ghost", lambda req: "ok")
        result = rt.run("let x = 1")
        assert result.diagnostics[0].line == 1

    def test_declared_but_unbacked_is_host_error(self) -> None:
        rt = WorkflowRuntime()  # no registration, no default agent
        result = rt.run('agent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "orphan" in msgs
        assert "backing" in msgs.lower()

    def test_declared_but_unbacked_diagnostic_reports_declaration_line(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let y = 1\nagent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        # The declaration is on line 2.
        assert result.diagnostics[0].line == 2

    def test_declared_and_registered_runs(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "output"

        rt = WorkflowRuntime()
        rt.register_agent("impl", agent)
        result = rt.run('agent impl\nlet x = impl "do it"')
        assert result.ok is True
        assert calls == ["do it"]

    def test_declared_with_default_agent_runs(self) -> None:
        # No dedicated registration, but a default agent backs the declared name.
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('agent any_name\nlet x = any_name "hi"')
        assert result.ok is True

    def test_both_error_categories_reported_together(self) -> None:
        rt = WorkflowRuntime()  # no default agent
        rt.register_agent("ghost", lambda req: "ok")
        # 'orphan' is declared but unbacked; 'ghost' is registered but undeclared.
        result = rt.run('agent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "ghost" in msgs
        assert "orphan" in msgs
        assert len(result.diagnostics) == 2

    def test_reconciliation_failure_skips_execution(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("ghost", agent)
        rt.run('print "side effect?"')
        assert calls == []
