"""Tests for the WorkflowRuntime shell (M0).

Covers:
- WorkflowRuntime constructor with default kwargs
- register_agent: duplicate rejection, reserved-name rejection (prompt, exec)
- run: returns a failed RunResult with a diagnostic when not implemented
- Diagnostic has .message (str) and .line (int)
- RunResult has .ok (bool), .diagnostics (list), .error (None for pre-exec failures)
- AglError and SourceSpan
- Token constants
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
        # M0: run() returns the not-implemented pre-execution failure.
        assert result.ok is False
        assert result.error is None
        assert result.diagnostics


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


class TestRunNotImplemented:
    def test_run_returns_run_result(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert isinstance(result, RunResult)

    def test_run_result_ok_is_false(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False

    def test_run_result_has_diagnostic(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert len(result.diagnostics) >= 1

    def test_run_result_diagnostic_has_message(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        diag = result.diagnostics[0]
        assert isinstance(diag.message, str)
        assert diag.message  # non-empty

    def test_run_result_diagnostic_has_line(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        diag = result.diagnostics[0]
        assert isinstance(diag.line, int)
        assert diag.line == 1

    def test_run_result_error_is_none(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        # pre-execution failure: error is None (no AgL exception was raised)
        assert result.error is None

    def test_run_with_inputs(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1", inputs={"key": "value"})
        assert isinstance(result, RunResult)
        assert result.ok is False

    def test_run_with_empty_inputs(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1", inputs={})
        assert isinstance(result, RunResult)

    def test_diagnostic_message_mentions_not_implemented(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        diag = result.diagnostics[0]
        # The message should mention that execution is not yet implemented
        assert "not implemented" in diag.message.lower() or "implementation" in diag.message.lower()


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
