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

        def agent(req: object) -> str:
            calls.append(req.prompt)  # type: ignore[attr-defined]
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
        received: list[object] = []

        def agent(req: object) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('let x = prompt "Hello world"')
        assert received[0].prompt == "Hello world"  # type: ignore[attr-defined]

    def test_request_agent_name_for_default(self) -> None:
        received: list[object] = []

        def agent(req: object) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('let x = prompt "Hi"')
        assert received[0].agent == "prompt"  # type: ignore[attr-defined]

    def test_request_agent_name_for_named(self) -> None:
        received: list[object] = []

        def reviewer(req: object) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this"')
        assert received[0].agent == "reviewer"  # type: ignore[attr-defined]


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
