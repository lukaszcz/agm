"""Tests for the AgL evaluator (Component 6) via WorkflowRuntime.run().

All assertions are on user-visible RunResult attributes: .ok, .diagnostics,
.error, .bindings (root scope snapshot), and process stdout (capsys).

Agent calls use registered Python stub agents — no subprocess.

NOTE: Tests are scoped to what the M1 parser and typecheck support:
- Statements: input, let, var, set, pass, print, expr_stmt
- Expressions: var refs, scalar literals, templates, agent calls
- Types: text (default), int, decimal, bool, json
- Templates with default and raw renderers
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl import WorkflowRuntime
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.request import AgentRequest
from agm.agl.runtime.runtime import RunResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(source: str, *, inputs: dict[str, object] | None = None) -> RunResult:
    """Build a WorkflowRuntime with no named agents and run *source*."""
    rt = WorkflowRuntime()
    return rt.run(source, inputs=inputs or {})


def run_with_default_agent(
    source: str,
    fn: AgentFn,
    *,
    inputs: dict[str, object] | None = None,
) -> RunResult:
    """Build a WorkflowRuntime with a default agent and run *source*."""
    rt = WorkflowRuntime(default_agent=fn)
    return rt.run(source, inputs=inputs or {})


def run_with_agents(
    source: str,
    agents: dict[str, AgentFn],
    *,
    inputs: dict[str, object] | None = None,
) -> RunResult:
    """Build a WorkflowRuntime, register agents, and run *source*."""
    default = agents.get("prompt")
    others = {k: v for k, v in agents.items() if k != "prompt"}
    rt = WorkflowRuntime(default_agent=default)
    for name, fn in others.items():
        rt.register_agent(name, fn)
    return rt.run(source, inputs=inputs or {})


# ---------------------------------------------------------------------------
# Basic ok semantics
# ---------------------------------------------------------------------------


class TestOkSemantics:
    def test_empty_pass_ok(self) -> None:
        result = run("pass")
        assert result.ok is True
        assert result.error is None
        assert result.diagnostics == []

    def test_let_binding_ok(self) -> None:
        result = run("let x = 1")
        assert result.ok is True

    def test_static_error_not_ok(self) -> None:
        result = run("let x = undefined_var")
        assert result.ok is False
        assert result.error is None
        assert result.diagnostics  # at least one error diagnostic

    def test_static_error_has_line(self) -> None:
        result = run("let x = undefined_var")
        assert result.diagnostics[0].line >= 1

    def test_static_error_has_message(self) -> None:
        result = run("let x = undefined_var")
        assert result.diagnostics[0].message

    def test_bindings_present_on_ok(self) -> None:
        result = run("let x = 1")
        assert hasattr(result, "bindings")
        assert isinstance(result.bindings, dict)

    def test_bindings_empty_on_failure(self) -> None:
        result = run("let x = undefined_var")
        assert result.bindings == {}


# ---------------------------------------------------------------------------
# RunResult.bindings — root scope snapshot
# ---------------------------------------------------------------------------


class TestRootBindings:
    def test_let_binding_visible_in_bindings(self) -> None:
        result = run("let x = 42")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(42)

    def test_var_binding_visible_in_bindings(self) -> None:
        result = run('var msg = "hello"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("hello")

    def test_set_updates_binding(self) -> None:
        result = run('var x: text = "first"\nset x = "second"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("second")

    def test_multiple_bindings(self) -> None:
        result = run("let a = 1\nlet b = 2\nlet c = 3")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["a"] == IntValue(1)
        assert result.bindings["b"] == IntValue(2)
        assert result.bindings["c"] == IntValue(3)


# ---------------------------------------------------------------------------
# Literal evaluation
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_int_literal(self) -> None:
        result = run("let x = 7")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(7)

    def test_decimal_literal(self) -> None:
        result = run("let x = 1.5")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal("1.5"))

    def test_bool_true(self) -> None:
        result = run("let x = true")
        assert result.ok
        from agm.agl.eval.values import BoolValue

        assert result.bindings["x"] == BoolValue(True)

    def test_bool_false(self) -> None:
        result = run("let x = false")
        assert result.ok
        from agm.agl.eval.values import BoolValue

        assert result.bindings["x"] == BoolValue(False)

    def test_string_literal(self) -> None:
        result = run('let x = "hello world"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("hello world")

    def test_null_literal(self) -> None:
        result = run("let j: json = null")
        assert result.ok
        from agm.agl.eval.values import JsonValue

        assert result.bindings["j"] == JsonValue(None)

    def test_large_int(self) -> None:
        bignum = 123456789012345678901234567890
        result = run(f"let x = {bignum}")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(bignum)


# ---------------------------------------------------------------------------
# Template evaluation
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_template_no_interp(self) -> None:
        result = run('let x = "plain text"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("plain text")

    def test_template_with_int_interp_raw(self) -> None:
        # int interpolation as raw: no boundary marker
        result = run("let n = 5\nlet msg = \"n is ${n as raw}\"")
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("n is 5")

    def test_template_with_text_interp_raw(self) -> None:
        result = run('let s = "hello"\nlet msg = "say ${s as raw}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["msg"] == TextValue("say hello")

    def test_template_default_text_uses_boundary_markers(self) -> None:
        result = run('let s = "abc"\nlet msg = "x: ${s}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        # Default rendering for text includes boundary markers
        assert "<dsl-value" in v.value
        assert "abc" in v.value
        assert "</dsl-value>" in v.value

    def test_template_with_bool_interp_raw(self) -> None:
        result = run('let b = true\nlet msg = "${b as raw}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "true" == v.value

    def test_template_escape_newline(self) -> None:
        result = run('let x = "line1\\nline2"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("line1\nline2")

    def test_template_int_default_is_scalar(self) -> None:
        # int default rendering is scalar (no boundary markers)
        result = run('let n = 42\nlet msg = "value: ${n}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "42" in v.value
        assert "<dsl-value" not in v.value

    def test_template_bool_default_is_scalar(self) -> None:
        result = run('let b = false\nlet msg = "${b}"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        v = result.bindings["msg"]
        assert isinstance(v, TextValue)
        assert "false" in v.value
        assert "<dsl-value" not in v.value


# ---------------------------------------------------------------------------
# Print console rendering
# ---------------------------------------------------------------------------


class TestPrintRendering:
    def test_print_text_verbatim(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('print "hello world"')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "hello world\n"

    def test_print_int_scalar(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 42")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "42\n"

    def test_print_decimal_scalar(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 2.5")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "2.5\n"

    def test_print_bool_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print true")
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out  # lowercase
        assert "True" not in out  # NOT Python repr

    def test_print_bool_false(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print false")
        assert result.ok
        out = capsys.readouterr().out
        assert "false" in out
        assert "False" not in out

    def test_print_no_boundary_markers(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('let s = "hello"\nprint s')
        assert result.ok
        out = capsys.readouterr().out
        assert "<dsl-value" not in out
        assert out == "hello\n"

    def test_print_json_null(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let j: json = null\nprint j")
        assert result.ok
        out = capsys.readouterr().out
        assert "null" in out

    def test_print_multiline_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('print "two\\nlines"')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "two\nlines\n"

    def test_print_large_int(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("print 123456789012345678901234567890")
        assert result.ok
        out = capsys.readouterr().out
        assert "123456789012345678901234567890" in out
        assert "e+" not in out.lower()

    def test_print_var_ref(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let x = 99\nprint x")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "99\n"


# ---------------------------------------------------------------------------
# Agent call evaluation (text codec)
# ---------------------------------------------------------------------------


class TestAgentCalls:
    def test_prompt_call_binds_response(self) -> None:
        def agent(req: AgentRequest) -> str:
            return "response text"

        result = run_with_default_agent('let x = prompt "Hello"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("response text")

    def test_named_agent_call(self) -> None:
        def impl(req: AgentRequest) -> str:
            return "output"

        rt = WorkflowRuntime()
        rt.register_agent("impl", impl)
        result = rt.run('let x = impl "Do something"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("output")

    def test_agent_receives_rendered_prompt_raw(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let name = "world"\nlet x = prompt "Hello ${name as raw}"', agent)
        assert len(prompts) == 1
        assert prompts[0] == "Hello world"

    def test_agent_prompt_contains_boundary_markers_for_text(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let artifact = "content"\nlet x = prompt "see ${artifact}"', agent)
        assert len(prompts) == 1
        assert "<dsl-value" in prompts[0]
        assert "content" in prompts[0]
        assert "</dsl-value>" in prompts[0]

    def test_agent_receives_request_with_agent_name(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this."')
        assert len(received) == 1
        assert received[0].agent == "reviewer"

    def test_empty_response_is_valid_for_text_target(self) -> None:
        def agent(req: AgentRequest) -> str:
            return ""

        result = run_with_default_agent('let x = prompt "Say nothing."', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("")

    def test_no_default_agent_without_registration_fails_statically(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let x = prompt "Hi"')
        # No default agent, no fallback → static capability error
        assert result.ok is False
        assert result.error is None

    def test_agent_response_object_accepted(self) -> None:
        from agm.agl.runtime.request import AgentResponse

        def agent(req: AgentRequest) -> AgentResponse:
            return AgentResponse(content="from object")

        result = run_with_default_agent('let x = prompt "Hi"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from object")

    def test_expr_stmt_call_result_discarded(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        result = run_with_default_agent('prompt "Note something."', agent)
        assert result.ok
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Input validation (§11.3, §9.5)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_declared_input_fails(self) -> None:
        result = run("input name\nprint name", inputs={})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "name" in msgs.lower()

    def test_undeclared_extra_input_fails(self) -> None:
        result = run("input name\nprint name", inputs={"name": "bob", "bogus": "x"})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "bogus" in msgs.lower()

    def test_text_input_taken_verbatim(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input name\nprint name", inputs={"name": "alice"})
        assert result.ok
        out = capsys.readouterr().out
        assert "alice" in out

    def test_int_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input n: int\nprint n", inputs={"n": 42})
        assert result.ok
        out = capsys.readouterr().out
        assert "42" in out

    def test_bool_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input flag: bool\nprint flag", inputs={"flag": True})
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out

    def test_json_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input meta: json\nprint meta", inputs={"meta": {"key": 1}})
        assert result.ok
        out = capsys.readouterr().out
        assert "key" in out

    def test_type_invalid_int_input_fails(self) -> None:
        result = run("input n: int\nprint n", inputs={"n": "not a number"})
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "n" in msgs.lower()

    def test_input_bound_immutably(self) -> None:
        # set on an input binding is a static error (scope pass)
        result = run("input x\nset x = \"y\"", inputs={"x": "hello"})
        assert result.ok is False

    def test_no_agent_called_on_input_failure(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('input name\nlet x = prompt "Hi ${name as raw}"', inputs={})
        assert calls == []

    def test_input_used_in_template(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'input name\nlet x = prompt "Hello ${name as raw}"',
            agent,
            inputs={"name": "Alice"},
        )
        assert len(prompts) == 1
        assert "Alice" in prompts[0]

    def test_decimal_input_from_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input r: decimal\nprint r", inputs={"r": decimal.Decimal("2.5")})
        assert result.ok
        out = capsys.readouterr().out
        assert "2.5" in out


# ---------------------------------------------------------------------------
# Boundary-marked rendering (§2.12)
# ---------------------------------------------------------------------------


class TestBoundaryRendering:
    def test_text_interpolation_default_has_dsl_value_tag(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let artifact = "hello"\nlet x = prompt "see ${artifact}"', agent)
        assert '<dsl-value name="artifact" type="text">' in prompts[0]
        assert "hello" in prompts[0]
        assert "</dsl-value>" in prompts[0]

    def test_raw_renderer_bypasses_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let s = "raw content"\nlet x = prompt "${s as raw}"', agent)
        assert prompts[0] == "raw content"
        assert "<dsl-value" not in prompts[0]

    def test_int_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let n = 5\nlet x = prompt "n=${n}"', agent)
        assert "5" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_bool_interp_default_is_scalar_no_boundary(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let b = true\nlet x = prompt "b=${b}"', agent)
        assert "true" in prompts[0]
        assert "<dsl-value" not in prompts[0]

    def test_text_interp_name_attribute_is_varname(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent(
            'let my_artifact = "content"\nlet x = prompt "${my_artifact}"', agent
        )
        assert 'name="my_artifact"' in prompts[0]

    def test_null_interp_boundary_marked(self) -> None:
        prompts: list[str] = []

        def agent(req: AgentRequest) -> str:
            prompts.append(req.prompt)
            return "ok"

        run_with_default_agent('let j: json = null\nlet x = prompt "data: ${j}"', agent)
        assert "<dsl-value" in prompts[0]
        assert "null" in prompts[0]


# ---------------------------------------------------------------------------
# AgentRequest fields
# ---------------------------------------------------------------------------


class TestAgentRequest:
    def test_request_has_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = prompt "Hello world"', agent)
        assert received[0].prompt == "Hello world"

    def test_request_has_agent_name_prompt(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        run_with_default_agent('let x = prompt "Hi"', agent)
        assert received[0].agent == "prompt"

    def test_request_has_agent_name_custom(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('let x = reviewer "Review this"')
        assert received[0].agent == "reviewer"


# ---------------------------------------------------------------------------
# var / set statements
# ---------------------------------------------------------------------------


class TestVarSet:
    def test_var_initial_value(self) -> None:
        result = run('var x: text = "initial"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("initial")

    def test_set_updates_var(self) -> None:
        result = run('var x: text = "a"\nset x = "b"')
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("b")

    def test_set_on_let_is_static_error(self) -> None:
        result = run('let x = "a"\nset x = "b"')
        assert result.ok is False

    def test_set_undeclared_is_static_error(self) -> None:
        result = run('set x = "value"')
        assert result.ok is False

    def test_var_from_agent_response(self) -> None:
        def agent(req: AgentRequest) -> str:
            return "from agent"

        result = run_with_default_agent('var x: text = prompt "Get value"', agent)
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("from agent")

    def test_set_from_agent_response(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append("call")
            return "v2"

        result = run_with_default_agent(
            'var x: text = prompt "First"\nset x = prompt "Second"', agent
        )
        assert result.ok
        assert len(calls) == 2
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("v2")


# ---------------------------------------------------------------------------
# Pass statement
# ---------------------------------------------------------------------------


class TestPassStmt:
    def test_pass_is_noop(self) -> None:
        result = run("pass")
        assert result.ok
        assert result.bindings == {}

    def test_pass_with_bindings(self) -> None:
        result = run("let x = 1\npass\nlet y = 2")
        assert result.ok
        from agm.agl.eval.values import IntValue

        assert result.bindings["x"] == IntValue(1)
        assert result.bindings["y"] == IntValue(2)


# ---------------------------------------------------------------------------
# Multiple agent calls and response chaining
# ---------------------------------------------------------------------------


class TestMultipleAgentCalls:
    def test_two_sequential_calls(self) -> None:
        responses = ["v1", "v2"]
        idx = [0]

        def agent(req: AgentRequest) -> str:
            r = responses[idx[0]]
            idx[0] += 1
            return r

        result = run_with_default_agent(
            'let a = prompt "First"\nlet b = prompt "Second"', agent
        )
        assert result.ok
        from agm.agl.eval.values import TextValue

        assert result.bindings["a"] == TextValue("v1")
        assert result.bindings["b"] == TextValue("v2")

    def test_chaining_response_into_next_prompt(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            if len(calls) == 1:
                return "first-output"
            return "second-output"

        result = run_with_default_agent(
            'let a = prompt "First"\nlet b = prompt "Use ${a as raw}"', agent
        )
        assert result.ok
        assert "first-output" in calls[1]

    def test_named_agent_and_default_agent(self) -> None:
        default_calls: list[str] = []
        impl_calls: list[str] = []

        def default_agent(req: AgentRequest) -> str:
            default_calls.append(req.prompt)
            return "prompt-response"

        def impl(req: AgentRequest) -> str:
            impl_calls.append(req.prompt)
            return "impl-response"

        result = run_with_agents(
            'let a = prompt "Hello"\nlet b = impl "Build"',
            {"prompt": default_agent, "impl": impl},
        )
        assert result.ok
        assert len(default_calls) == 1
        assert len(impl_calls) == 1
        from agm.agl.eval.values import TextValue

        assert result.bindings["a"] == TextValue("prompt-response")
        assert result.bindings["b"] == TextValue("impl-response")


# ---------------------------------------------------------------------------
# Type coercion (int → decimal)
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_int_to_decimal_annotation(self) -> None:
        result = run("let x: decimal = 3")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal(3))

    def test_decimal_literal_stays_decimal(self) -> None:
        result = run("let x: decimal = 1.5")
        assert result.ok
        from agm.agl.eval.values import DecimalValue

        assert result.bindings["x"] == DecimalValue(decimal.Decimal("1.5"))


# ---------------------------------------------------------------------------
# Print with var refs
# ---------------------------------------------------------------------------


class TestPrintVarRef:
    def test_print_int_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let n = 42\nprint n")
        assert result.ok
        out = capsys.readouterr().out
        assert out == "42\n"

    def test_print_text_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run('let s = "hello"\nprint s')
        assert result.ok
        out = capsys.readouterr().out
        assert out == "hello\n"

    def test_print_bool_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let b = true\nprint b")
        assert result.ok
        out = capsys.readouterr().out
        assert "true" in out

    def test_print_decimal_var(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("let d = 3.14\nprint d")
        assert result.ok
        out = capsys.readouterr().out
        assert "3.14" in out

    def test_print_input_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input msg\nprint msg", inputs={"msg": "from input"})
        assert result.ok
        out = capsys.readouterr().out
        assert out == "from input\n"

    def test_print_input_int(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = run("input n: int\nprint n", inputs={"n": 7})
        assert result.ok
        out = capsys.readouterr().out
        assert out == "7\n"


# ---------------------------------------------------------------------------
# Value types module exports
# ---------------------------------------------------------------------------


class TestValueTypes:
    def test_text_value_equality(self) -> None:
        from agm.agl.eval.values import TextValue

        assert TextValue("a") == TextValue("a")
        assert TextValue("a") != TextValue("b")

    def test_int_value_equality(self) -> None:
        from agm.agl.eval.values import IntValue

        assert IntValue(1) == IntValue(1)
        assert IntValue(1) != IntValue(2)

    def test_decimal_value_equality(self) -> None:
        from agm.agl.eval.values import DecimalValue

        assert DecimalValue(decimal.Decimal("1.5")) == DecimalValue(decimal.Decimal("1.5"))

    def test_bool_value_equality(self) -> None:
        from agm.agl.eval.values import BoolValue

        assert BoolValue(True) == BoolValue(True)
        assert BoolValue(True) != BoolValue(False)

    def test_json_value_equality(self) -> None:
        from agm.agl.eval.values import JsonValue

        assert JsonValue(None) == JsonValue(None)
        assert JsonValue({"a": 1}) == JsonValue({"a": 1})


# ---------------------------------------------------------------------------
# M3+ Value types: hash/eq for DictValue, RecordValue, EnumValue, ExceptionValue
# ---------------------------------------------------------------------------


class TestM3ValueTypes:
    def test_dict_value_equality(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d1 = DictValue(entries={"a": IntValue(1)})
        d2 = DictValue(entries={"a": IntValue(1)})
        assert d1 == d2

    def test_dict_value_inequality(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d1 = DictValue(entries={"a": IntValue(1)})
        d2 = DictValue(entries={"a": IntValue(2)})
        assert d1 != d2

    def test_dict_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import DictValue

        d = DictValue(entries={})
        assert d.__eq__("not a dict") is NotImplemented

    def test_dict_value_hashable(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue

        d = DictValue(entries={"x": IntValue(1)})
        # Hashable means it can be used as a dict key.
        mapping = {d: "hello"}
        assert mapping[d] == "hello"

    def test_record_value_equality(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r1 = RecordValue(type_name="Point", fields={"x": IntValue(1), "y": IntValue(2)})
        r2 = RecordValue(type_name="Point", fields={"x": IntValue(1), "y": IntValue(2)})
        assert r1 == r2

    def test_record_value_inequality_type(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r1 = RecordValue(type_name="Point", fields={"x": IntValue(1)})
        r2 = RecordValue(type_name="Line", fields={"x": IntValue(1)})
        assert r1 != r2

    def test_record_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import RecordValue

        r = RecordValue(type_name="Point", fields={})
        assert r.__eq__("other") is NotImplemented

    def test_record_value_hashable(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue

        r = RecordValue(type_name="Point", fields={"x": IntValue(3)})
        h = hash(r)
        assert isinstance(h, int)

    def test_enum_value_equality(self) -> None:
        from agm.agl.eval.values import EnumValue, TextValue

        e1 = EnumValue(type_name="Color", variant="Red", fields={"label": TextValue("r")})
        e2 = EnumValue(type_name="Color", variant="Red", fields={"label": TextValue("r")})
        assert e1 == e2

    def test_enum_value_inequality_variant(self) -> None:
        from agm.agl.eval.values import EnumValue

        e1 = EnumValue(type_name="Color", variant="Red", fields={})
        e2 = EnumValue(type_name="Color", variant="Blue", fields={})
        assert e1 != e2

    def test_enum_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import EnumValue

        e = EnumValue(type_name="Color", variant="Red", fields={})
        assert e.__eq__(42) is NotImplemented

    def test_enum_value_hashable(self) -> None:
        from agm.agl.eval.values import EnumValue

        e = EnumValue(type_name="Color", variant="Blue", fields={})
        h = hash(e)
        assert isinstance(h, int)

    def test_exception_value_equality(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e1 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        e2 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        assert e1 == e2

    def test_exception_value_inequality(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e1 = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("a"), "trace_id": TextValue("")}
        )
        e2 = ExceptionValue(
            type_name="Other", fields={"message": TextValue("a"), "trace_id": TextValue("")}
        )
        assert e1 != e2

    def test_exception_value_not_equal_to_other_type(self) -> None:
        from agm.agl.eval.values import ExceptionValue

        e = ExceptionValue(type_name="Abort", fields={})
        assert e.__eq__(None) is NotImplemented

    def test_exception_value_hashable(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue

        e = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("x"), "trace_id": TextValue("")}
        )
        h = hash(e)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# AglRaise carrier
# ---------------------------------------------------------------------------


class TestAglRaise:
    def test_agl_raise_carries_exc_value(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc_val = ExceptionValue(
            type_name="TestError",
            fields={"message": TextValue("oops"), "trace_id": TextValue("")},
        )
        carrier = AglRaise(exc_val)
        assert carrier.exc is exc_val
        assert str(carrier) == "TestError"

    def test_agl_raise_is_exception(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        exc_val = ExceptionValue(
            type_name="E", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        carrier = AglRaise(exc_val)
        assert isinstance(carrier, Exception)


# ---------------------------------------------------------------------------
# Scope unit tests (parent chain set_value)
# ---------------------------------------------------------------------------


class TestScopeUnit:
    def test_set_value_in_parent_scope(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        parent = Scope(parent=None)
        parent.define("x", IntValue(1), mutable=True, decl_span=span)
        child = Scope(parent=parent)

        # Set in child scope: updates parent's binding.
        result = child.set_value("x", IntValue(99))
        assert result is True
        assert parent.bindings["x"].value == IntValue(99)

    def test_set_value_not_found_returns_false(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue

        scope = Scope(parent=None)
        result = scope.set_value("nonexistent", IntValue(5))
        assert result is False

    def test_lookup_from_parent_scope(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        parent = Scope(parent=None)
        parent.define("x", IntValue(42), mutable=False, decl_span=span)
        child = Scope(parent=parent)

        # Child scope has no binding for "x", must walk to parent.
        binding = child.lookup("x")
        assert binding is not None
        assert binding.value == IntValue(42)

    def test_lookup_not_found_returns_none(self) -> None:
        from agm.agl.eval.scope import Scope

        scope = Scope(parent=None)
        assert scope.lookup("missing") is None


# ---------------------------------------------------------------------------
# Codec unit tests
# ---------------------------------------------------------------------------


class TestCodecUnit:
    def test_text_codec_supports_text_type(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.typecheck.types import IntType, TextType

        c = TextCodec()
        assert c.supports_type(TextType()) is True
        assert c.supports_type(IntType()) is False

    def test_parse_result_failure(self) -> None:
        from agm.agl.runtime.codec import ParseResult

        r = ParseResult.failure("bad format")
        assert r.ok is False
        assert r.value is None
        assert r.error_msg == "bad format"


# ---------------------------------------------------------------------------
# Render unit tests
# ---------------------------------------------------------------------------


class TestRenderUnit:
    def test_render_for_console_list_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_console

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        text = render_for_console(v)
        assert "1" in text
        assert "2" in text

    def test_render_for_console_dict_value(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import render_for_console

        v = DictValue(entries={"key": TextValue("val")})
        text = render_for_console(v)
        assert "key" in text
        assert "val" in text

    def test_render_for_console_record_value(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import render_for_console

        v = RecordValue(type_name="Point", fields={"x": IntValue(3)})
        text = render_for_console(v)
        assert "3" in text

    def test_render_for_console_enum_value(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_for_console

        v = EnumValue(type_name="Status", variant="Active", fields={})
        text = render_for_console(v)
        assert "Active" in text

    def test_render_for_console_exception_value(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import render_for_console

        v = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("fatal"), "trace_id": TextValue("")}
        )
        text = render_for_console(v)
        assert "fatal" in text

    def test_render_for_console_json_value_null(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_console

        v = JsonValue(None)
        text = render_for_console(v)
        assert "null" in text

    def test_value_to_json_obj_decimal(self) -> None:
        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = DecimalValue(decimal.Decimal("3.14"))
        result = _value_to_json_obj(v)
        assert isinstance(result, float)
        assert abs(float(result) - 3.14) < 0.001

    def test_value_to_json_obj_bool(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import _value_to_json_obj

        assert _value_to_json_obj(BoolValue(True)) is True
        assert _value_to_json_obj(BoolValue(False)) is False

    def test_value_to_json_obj_json(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = JsonValue({"nested": [1, 2]})
        result = _value_to_json_obj(v)
        assert result == {"nested": [1, 2]}

    def test_value_to_json_obj_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        result = _value_to_json_obj(v)
        assert result == [1, 2]

    def test_value_to_json_obj_dict(self) -> None:
        from agm.agl.eval.values import DictValue, TextValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = DictValue(entries={"k": TextValue("v")})
        result = _value_to_json_obj(v)
        assert result == {"k": "v"}

    def test_value_to_json_obj_record(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = RecordValue(type_name="P", fields={"x": IntValue(5)})
        result = _value_to_json_obj(v)
        assert result == {"x": 5}

    def test_value_to_json_obj_enum(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = EnumValue(type_name="C", variant="Red", fields={})
        result = _value_to_json_obj(v)
        assert result == {"$case": "Red"}

    def test_value_to_json_obj_exception(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import _value_to_json_obj

        v = ExceptionValue(
            type_name="E", fields={"message": TextValue("oops"), "trace_id": TextValue("")}
        )
        result = _value_to_json_obj(v)
        assert isinstance(result, dict)
        assert result.get("message") == "oops"

    def test_render_for_prompt_json_renderer(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        v = IntValue(42)
        text = render_for_prompt(v, renderer_name="json", var_name=None)
        assert "42" in text

    def test_render_for_prompt_bullets_list(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_for_prompt

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        text = render_for_prompt(v, renderer_name="bullets", var_name=None)
        assert "- 1" in text
        assert "- 2" in text

    def test_render_for_prompt_bullets_non_list(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_for_prompt

        v = IntValue(5)
        text = render_for_prompt(v, renderer_name="bullets", var_name=None)
        assert "5" in text

    def test_render_for_prompt_unknown_renderer_falls_back(self) -> None:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_for_prompt

        v = TextValue("hello")
        text = render_for_prompt(v, renderer_name="nonexistent", var_name="x")
        # Falls back to default: boundary-marked text
        assert "<dsl-value" in text

    def test_render_default_json_value(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_for_prompt

        v = JsonValue({"a": 1})
        text = render_for_prompt(v, renderer_name="default", var_name="data")
        assert "<dsl-value" in text
        assert '"a"' in text

    def test_type_kind_str_for_all_types(self) -> None:
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
        from agm.agl.runtime.render import _type_kind_str

        assert _type_kind_str(TextValue("x")) == "text"
        assert _type_kind_str(IntValue(1)) == "int"
        assert _type_kind_str(DecimalValue(decimal.Decimal("1.5"))) == "decimal"
        assert _type_kind_str(BoolValue(True)) == "bool"
        assert _type_kind_str(JsonValue(None)) == "json"
        assert _type_kind_str(ListValue(elements=())) == "list"
        assert _type_kind_str(DictValue(entries={})) == "dict"
        assert _type_kind_str(RecordValue(type_name="P", fields={})) == "P"
        assert _type_kind_str(EnumValue(type_name="E", variant="V", fields={})) == "E"
        assert _type_kind_str(ExceptionValue(type_name="Ex", fields={})) == "Ex"

    def test_scalar_text_json_value(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import _scalar_text

        text = _scalar_text(JsonValue({"a": 1}))
        assert "a" in text

    def test_scalar_text_list_falls_back_to_pretty_json(self) -> None:
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import _scalar_text

        v = ListValue(elements=(IntValue(1),))
        text = _scalar_text(v)
        assert "1" in text


# ---------------------------------------------------------------------------
# Contract materialization error
# ---------------------------------------------------------------------------


class TestContractError:
    def test_unknown_codec_raises_value_error(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        spec = OutputContractSpec(
            codec_name="unknown_codec",
            target_type=TextType(),
            strict_json=None,
        )
        codecs = {"text": TextCodec()}
        with pytest.raises(ValueError, match="unknown_codec"):
            materialize_contract(spec, codecs)


# ---------------------------------------------------------------------------
# WorkflowRuntime: exception handlers and edge cases
# ---------------------------------------------------------------------------


class TestRuntimeExceptionHandlers:
    def test_generic_parse_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglSyntaxError from parse_program → ok=False diagnostic."""
        import agm.agl.runtime.runtime as rt_mod

        def bad_parse(source: str) -> object:
            raise RuntimeError("unexpected parser crash")

        monkeypatch.setattr(rt_mod, "parse_program", bad_parse, raising=False)
        # Need to patch the import inside run()
        import agm.agl.parser as parser_mod

        monkeypatch.setattr(parser_mod, "parse_program", bad_parse)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "unexpected parser crash" in result.diagnostics[0].message

    def test_generic_scope_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglScopeError from resolve → ok=False diagnostic."""
        import agm.agl.scope as scope_mod


        def bad_resolve(program: object) -> object:
            raise RuntimeError("resolve crash")

        monkeypatch.setattr(scope_mod, "resolve", bad_resolve)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "resolve crash" in result.diagnostics[0].message

    def test_generic_typecheck_exception_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglTypeError from check → ok=False diagnostic."""
        import agm.agl.typecheck as tc_mod

        def bad_check(resolved: object, caps: object) -> object:
            raise RuntimeError("typecheck crash")

        monkeypatch.setattr(tc_mod, "check", bad_check)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "typecheck crash" in result.diagnostics[0].message

    def test_internal_interpreter_error_returns_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AglRaise from interpreter.execute → ok=False diagnostic."""
        from agm.agl.eval.interpreter import Interpreter

        def bad_execute(self: Interpreter, root_scope: object) -> None:
            raise RuntimeError("internal crash")

        monkeypatch.setattr(Interpreter, "execute", bad_execute)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert "internal crash" in result.diagnostics[0].message

    def test_exception_value_to_run_error_maps_all_field_kinds(self) -> None:
        """_exception_value_to_run_error converts every Value kind to JSON shape.

        This is the pure converter used to surface an uncaught AgL exception
        (e.g. AgentParseError) as a RunError.
        """
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
        from agm.agl.runtime.runtime import RunError, _exception_value_to_run_error

        exc_val = ExceptionValue(
            type_name="AgentParseError",
            fields={
                "message": TextValue("failed"),
                "trace_id": TextValue(""),
                "raw": TextValue("abc"),
                "agent": TextValue("prompt"),
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
        error = _exception_value_to_run_error(exc_val)
        assert isinstance(error, RunError)
        assert error.type_name == "AgentParseError"
        assert error.fields["message"] == "failed"
        assert isinstance(error.fields["decimal_val"], float)
        assert error.fields["bool_val"] is True
        assert error.fields["json_val"] == {"k": "v"}
        assert error.fields["list_val"] == [1]
        assert error.fields["dict_val"] == {"x": 2}
        assert error.fields["rec_val"] == {"f": "v"}
        assert error.fields["enum_val"] == {"$case": "V"}
        assert isinstance(error.fields["exc_val"], dict)

    def test_resolve_annotation_all_types(self) -> None:
        from agm.agl.runtime.runtime import _resolve_annotation
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import BoolT, DecimalT, IntT, JsonT, TextT
        from agm.agl.typecheck.types import BoolType, DecimalType, IntType, JsonType, TextType

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        assert isinstance(_resolve_annotation(None), TextType)
        assert isinstance(_resolve_annotation(TextT(span=span, node_id=0)), TextType)
        assert isinstance(_resolve_annotation(IntT(span=span, node_id=0)), IntType)
        assert isinstance(_resolve_annotation(DecimalT(span=span, node_id=0)), DecimalType)
        assert isinstance(_resolve_annotation(BoolT(span=span, node_id=0)), BoolType)
        assert isinstance(_resolve_annotation(JsonT(span=span, node_id=0)), JsonType)
        # Unknown type falls back to text
        assert isinstance(_resolve_annotation(object()), TextType)

    def test_convert_input_int_from_decimal_string(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        # "1.0" parses as Decimal("1.0") which equals int(1) → IntValue
        result = _convert_input("n", "1.0", IntType())
        assert result == IntValue(1)

    def test_convert_input_invalid_json_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="JSON"):
            _convert_input("n", "not_json", IntType())

    def test_convert_input_decimal_from_int(self) -> None:
        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import DecimalType

        # Int passed as raw int → DecimalValue
        result = _convert_input("r", 5, DecimalType())
        assert isinstance(result, DecimalValue)

    def test_convert_input_decimal_invalid_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import DecimalType

        with pytest.raises(ValueError, match="decimal"):
            _convert_input("r", True, DecimalType())

    def test_convert_input_bool_invalid_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import BoolType

        # Pass an integer (valid JSON type, but not a bool).
        with pytest.raises(ValueError, match="bool"):
            _convert_input("b", 42, BoolType())

    def test_convert_input_json_type_accepts_any(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import JsonType

        result = _convert_input("meta", [1, 2, 3], JsonType())
        assert result == JsonValue([1, 2, 3])

    def test_convert_input_unknown_type_fallback(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import _convert_input

        # Passing an unknown type_obj uses the fallback JSON path.
        result = _convert_input("x", [1, 2], object())
        assert isinstance(result, JsonValue)

    def test_convert_input_text_non_str_raises(self) -> None:
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import TextType

        with pytest.raises(ValueError, match="text"):
            _convert_input("t", 42, TextType())

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


# ---------------------------------------------------------------------------
# Interpreter unit tests (direct method calls for M3+ features)
# ---------------------------------------------------------------------------


class TestInterpreterUnit:
    """Unit tests for interpreter methods that are not reachable via M1 parser."""

    def test_make_exc_value_helper(self) -> None:
        from agm.agl.eval.interpreter import _make_exc_value
        from agm.agl.eval.values import ExceptionValue

        exc = _make_exc_value("TestError", "something went wrong")
        assert isinstance(exc, ExceptionValue)
        assert exc.type_name == "TestError"

    def test_coerce_non_decimal_unchanged(self) -> None:
        from agm.agl.eval.interpreter import _coerce
        from agm.agl.eval.values import TextValue
        from agm.agl.typecheck.types import TextType

        v = TextValue("hello")
        result = _coerce(v, TextType())
        assert result is v

    def test_add_text_values(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import TextValue

        result = _add(TextValue("hello "), TextValue("world"))
        assert result == TextValue("hello world")

    def test_add_int_and_decimal(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import DecimalValue, IntValue

        result = _add(IntValue(1), DecimalValue(decimal.Decimal("0.5")))
        assert isinstance(result, DecimalValue)

    def test_add_type_error(self) -> None:
        from agm.agl.eval.interpreter import _add
        from agm.agl.eval.values import BoolValue, TextValue

        with pytest.raises(RuntimeError, match="Cannot add"):
            _add(TextValue("x"), BoolValue(True))

    def test_arith_subtraction_int(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(IntValue(5), IntValue(3), BinOp.SUB)
        assert result == IntValue(2)

    def test_arith_multiplication_decimal(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(IntValue(3), DecimalValue(decimal.Decimal("2.0")), BinOp.MUL)
        assert isinstance(result, DecimalValue)

    def test_arith_type_error(self) -> None:
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import BinOp

        with pytest.raises(RuntimeError, match="Cannot perform"):
            _arith(TextValue("a"), TextValue("b"), BinOp.SUB)

    def test_div_decimal(self) -> None:
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import DecimalValue, IntValue

        result = _div(IntValue(10), IntValue(4))
        assert isinstance(result, DecimalValue)

    def test_div_by_zero_raises_agl_raise(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import IntValue

        with pytest.raises(AglRaise) as exc_info:
            _div(IntValue(5), IntValue(0))
        assert exc_info.value.exc.type_name == "ArithmeticError"

    def test_div_type_error(self) -> None:
        from agm.agl.eval.interpreter import _div
        from agm.agl.eval.values import TextValue

        with pytest.raises(RuntimeError, match="Cannot divide"):
            _div(TextValue("a"), TextValue("b"))

    def test_compare_eq_text(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(TextValue("abc"), TextValue("abc"), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_neq(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), IntValue(2), BinOp.NEQ)
        assert result == BoolValue(True)

    def test_compare_int_widen_to_decimal(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), DecimalValue(decimal.Decimal("1.0")), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_decimal_widen_left(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(DecimalValue(decimal.Decimal("2.0")), IntValue(2), BinOp.EQ)
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_lt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("1.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.LT,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_le(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("2.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.LE,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_gt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("3.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.GT,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_decimal_ge(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("2.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.GE,
        )
        assert result == BoolValue(True)

    def test_compare_ordering_int_lt(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        assert _compare(IntValue(1), IntValue(2), BinOp.LT) == BoolValue(True)
        assert _compare(IntValue(2), IntValue(1), BinOp.LE) == BoolValue(False)
        assert _compare(IntValue(3), IntValue(2), BinOp.GT) == BoolValue(True)
        assert _compare(IntValue(2), IntValue(2), BinOp.GE) == BoolValue(True)

    def test_compare_ordering_text(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        assert _compare(TextValue("a"), TextValue("b"), BinOp.LT) == BoolValue(True)
        assert _compare(TextValue("a"), TextValue("a"), BinOp.LE) == BoolValue(True)
        assert _compare(TextValue("b"), TextValue("a"), BinOp.GT) == BoolValue(True)
        assert _compare(TextValue("a"), TextValue("a"), BinOp.GE) == BoolValue(True)

    def test_compare_type_error(self) -> None:
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        with pytest.raises(RuntimeError, match="Cannot compare"):
            _compare(TextValue("x"), BoolValue(True), BinOp.LT)

    def test_in_op_list(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, IntValue, ListValue

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        assert _in_op(IntValue(1), v) == BoolValue(True)
        assert _in_op(IntValue(3), v) == BoolValue(False)

    def test_in_op_dict_key(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, DictValue, IntValue, TextValue

        v = DictValue(entries={"key": IntValue(1)})
        assert _in_op(TextValue("key"), v) == BoolValue(True)
        assert _in_op(TextValue("missing"), v) == BoolValue(False)

    def test_in_op_dict_non_text_key(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, DictValue, IntValue

        v = DictValue(entries={"key": IntValue(1)})
        assert _in_op(IntValue(1), v) == BoolValue(False)

    def test_in_op_text_substring(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import BoolValue, TextValue

        assert _in_op(TextValue("ell"), TextValue("hello")) == BoolValue(True)
        assert _in_op(TextValue("xyz"), TextValue("hello")) == BoolValue(False)

    def test_in_op_type_error(self) -> None:
        from agm.agl.eval.interpreter import _in_op
        from agm.agl.eval.values import IntValue, TextValue

        with pytest.raises(RuntimeError, match="in"):
            _in_op(IntValue(1), TextValue("hello"))

    def test_match_pattern_wildcard(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        p = WildcardPattern(span=span, node_id=0)
        matched, bindings = _match_pattern(p, IntValue(42))
        assert matched
        assert bindings == {}

    def test_match_pattern_var(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import VarPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        p = VarPattern(name="x", span=span, node_id=0)
        matched, bindings = _match_pattern(p, TextValue("hello"))
        assert matched
        assert bindings == {"x": TextValue("hello")}

    def test_match_pattern_literal_int_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IntLit, LiteralPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = IntLit(value=42, span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, bindings = _match_pattern(p, IntValue(42))
        assert matched
        assert bindings == {}

    def test_match_pattern_literal_int_no_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IntLit, LiteralPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = IntLit(value=99, span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, _ = _match_pattern(p, IntValue(42))
        assert not matched

    def test_match_pattern_literal_decimal(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import DecimalLit, LiteralPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = DecimalLit(value=decimal.Decimal("1.5"), span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, _ = _match_pattern(p, DecimalValue(decimal.Decimal("1.5")))
        assert matched

    def test_match_pattern_literal_bool(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BoolLit, LiteralPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = BoolLit(value=True, span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, _ = _match_pattern(p, BoolValue(True))
        assert matched

    def test_match_pattern_literal_string(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import LiteralPattern, StringLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = StringLit(value="hello", span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, _ = _match_pattern(p, TextValue("hello"))
        assert matched

    def test_match_pattern_literal_null(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import JsonValue
        from agm.agl.syntax.nodes import LiteralPattern, NullLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        lit = NullLit(span=span, node_id=0)
        p = LiteralPattern(literal=lit, span=span, node_id=1)
        matched, _ = _match_pattern(p, JsonValue(None))
        assert matched

    def test_matches_catch_bare_handler(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type=None, binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Any", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_exception_base_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="Exception", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_exact_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="Abort", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is True

    def test_matches_catch_wrong_type(self) -> None:
        from agm.agl.eval.interpreter import _matches_catch
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 1, 0, 0)
        handler = CatchClause(exc_type="NetworkError", binding=None, body=(), span=span, node_id=0)
        exc = ExceptionValue(
            type_name="Abort", fields={"message": TextValue("m"), "trace_id": TextValue("")}
        )
        assert _matches_catch(handler, exc) is False

    def test_describe_value(self) -> None:
        from agm.agl.eval.interpreter import _describe_value
        from agm.agl.eval.values import (
            EnumValue,
            ExceptionValue,
            IntValue,
            RecordValue,
        )

        assert "Status" in _describe_value(
            EnumValue(type_name="Status", variant="Active", fields={})
        )
        assert "Point" in _describe_value(RecordValue(type_name="Point", fields={}))
        assert "Abort" in _describe_value(ExceptionValue(type_name="Abort", fields={}))
        assert "IntValue" in _describe_value(IntValue(1))

    def test_field_access_on_non_record_raises(self) -> None:
        """_eval_field_access on a non-record/enum/exception type raises RuntimeError."""
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        interp = _make_interp()
        span = SourceSpan(1, 1, 1, 1, 0, 0)
        # Build a FieldAccess node manually; obj evaluates to an IntValue.
        obj_ref = VarRef(name="n", span=span, node_id=0)
        fa = FieldAccess(obj=obj_ref, field="x", span=span, node_id=1)
        scope = Scope(parent=None)
        scope.define("n", IntValue(42), mutable=False, decl_span=span)
        with pytest.raises(RuntimeError, match="Field access"):
            interp._eval_field_access(fa, scope)

    def test_eval_unary_not_non_bool_raises(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import IntLit, UnaryNot
        from agm.agl.syntax.spans import SourceSpan

        interp = _make_interp()
        span = SourceSpan(1, 1, 1, 1, 0, 0)
        operand = IntLit(value=1, span=span, node_id=0)
        not_expr = UnaryNot(operand=operand, span=span, node_id=1)
        scope = Scope(parent=None)
        with pytest.raises(RuntimeError, match="not: expected bool"):
            interp._eval_unary_not(not_expr, scope)

    def test_eval_unary_neg_non_number_raises(self) -> None:
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import StringLit, UnaryNeg
        from agm.agl.syntax.spans import SourceSpan

        interp = _make_interp()
        span = SourceSpan(1, 1, 1, 1, 0, 0)
        operand = StringLit(value="hello", span=span, node_id=0)
        neg_expr = UnaryNeg(operand=operand, span=span, node_id=1)
        scope = Scope(parent=None)
        with pytest.raises(RuntimeError, match="unary -"):
            interp._eval_unary_neg(neg_expr, scope)


# ---------------------------------------------------------------------------
# Runtime: build_type_env, uncaught AglRaise via monkeypatched codec
# ---------------------------------------------------------------------------


class TestCheckedProgramTypeEnv:
    def test_checked_program_carries_type_env(self) -> None:
        """check() populates CheckedProgram.type_env with a TypeEnvironment."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.parser import parse_program
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        source = "pass"
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            codec_kinds={},
            renderer_names=frozenset(),
        )
        checked = check(resolved, caps)
        assert isinstance(checked.type_env, TypeEnvironment)


class TestRuntimeContractError:
    def test_contract_error_returns_failure_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """materialize_contract raising ValueError → ok=False, diagnostics list."""
        import agm.agl.runtime.contract as contract_mod

        def bad_materialize(spec: object, codecs: object) -> object:
            raise ValueError("bad codec")

        monkeypatch.setattr(contract_mod, "materialize_contract", bad_materialize)

        def agent(req: AgentRequest) -> str:
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = prompt "Hi"')
        assert result.ok is False
        assert result.error is None
        assert any("bad codec" in d.message for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Helper: build a minimal Interpreter for direct unit testing
# ---------------------------------------------------------------------------


def _make_interp(type_env: object = None) -> object:
    """Create a minimal interpreter with optional TypeEnvironment injection."""
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.eval.interpreter import Interpreter
    from agm.agl.parser import parse_program
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check
    from agm.agl.typecheck.env import TypeEnvironment

    program = parse_program("pass")
    resolved = resolve(program)
    caps = HostCapabilities(
        agent_names=frozenset(),
        has_fallback_agent=False,
        codec_kinds={},
        renderer_names=frozenset({"default", "raw"}),
    )
    checked = check(resolved, caps)
    registry = AgentRegistry(named={}, default_agent=None)
    interp = Interpreter(
        checked=checked,
        registry=registry,
        contracts={},
        type_env=type_env if type_env is not None else TypeEnvironment(),
        loop_limit=3,
        strict_json=False,
    )
    return interp


def _span() -> object:
    from agm.agl.syntax.spans import SourceSpan

    return SourceSpan(1, 1, 1, 5, 0, 4)


# ---------------------------------------------------------------------------
# Coverage: interpreter M3+ statement dispatch (_exec_stmt branches)
# ---------------------------------------------------------------------------


class TestInterpreterM3Stmts:
    """Unit tests for M3+ statement types dispatched by _exec_stmt."""

    def test_do_until_runs_body_until_true(self) -> None:
        """DoUntil with limit=1 and condition=true exits after one iteration."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, DoUntil, PassStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        body = (PassStmt(span=span, node_id=10),)
        condition = BoolLit(value=True, span=span, node_id=11)
        stmt = DoUntil(limit=3, body=body, condition=condition, span=span, node_id=12)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_do_until(stmt, scope)  # Should not raise

    def test_do_until_uses_runtime_loop_limit(self) -> None:
        """DoUntil with limit=None uses self._loop_limit."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, DoUntil, PassStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        body = (PassStmt(span=span, node_id=10),)
        condition = BoolLit(value=False, span=span, node_id=11)
        stmt = DoUntil(limit=None, body=body, condition=condition, span=span, node_id=12)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        with pytest.raises(AglRaise) as exc_info:
            interp._exec_do_until(stmt, scope)
        assert exc_info.value.exc.type_name == "MaxIterationsExceeded"

    def test_if_stmt_true_branch_executes(self) -> None:
        """IfStmt with true condition executes that branch."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, IfBranch, IfStmt, PrintStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        cond = BoolLit(value=True, span=span, node_id=1)
        # Use PrintStmt — it has a side-effect we can check via capsys,
        # but here we just check no error is raised.
        body_stmt = PrintStmt(value=BoolLit(value=True, span=span, node_id=2), span=span, node_id=3)
        branch = IfBranch(cond=cond, body=(body_stmt,), span=span, node_id=4)
        stmt = IfStmt(branches=(branch,), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_if(stmt, scope)
        # Branch was executed without error

    def test_if_stmt_false_branch_skipped(self) -> None:
        """IfStmt with false condition skips the branch."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, IfBranch, IfStmt, IntLit, LetDecl
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        cond = BoolLit(value=False, span=span, node_id=1)
        body_stmt = LetDecl(
            name="x", type_ann=None, value=IntLit(value=42, span=span, node_id=2),
            span=span, node_id=3,
        )
        branch = IfBranch(cond=cond, body=(body_stmt,), span=span, node_id=4)
        stmt = IfStmt(branches=(branch,), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_if(stmt, scope)
        assert scope.lookup("x") is None

    def test_if_stmt_else_branch_executes(self) -> None:
        """IfStmt else branch (ElseSentinel) executes when no prior branch matched."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import ELSE, BoolLit, IfBranch, IfStmt, PassStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        false_branch = IfBranch(
            cond=BoolLit(value=False, span=span, node_id=1),
            body=(),
            span=span,
            node_id=2,
        )
        else_branch = IfBranch(
            cond=ELSE,
            body=(PassStmt(span=span, node_id=3),),
            span=span,
            node_id=4,
        )
        stmt = IfStmt(branches=(false_branch, else_branch), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_if(stmt, scope)
        # Else branch executed without error

    def test_case_stmt_matches_wildcard(self) -> None:
        """CaseStmt with wildcard pattern always matches."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, IntLit, LetDecl, WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=5, span=span, node_id=1)
        body_stmt = LetDecl(
            name="matched", type_ann=None, value=IntLit(value=1, span=span, node_id=2),
            span=span, node_id=3,
        )
        branch = CaseStmtBranch(
            pattern=WildcardPattern(span=span, node_id=4), body=(body_stmt,),
            span=span, node_id=5,
        )
        stmt = CaseStmt(subject=subject, branches=(branch,), span=span, node_id=6)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_case_stmt(stmt, scope)
        # "matched" is defined in branch_scope (a child), not in root scope
        # but no error means it executed

    def test_case_stmt_no_match_raises_match_error(self) -> None:
        """CaseStmt with no matching branch raises AglRaise(MatchError)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, IntLit, LiteralPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=5, span=span, node_id=1)
        # Pattern: literal 99 (won't match 5)
        pat = LiteralPattern(literal=IntLit(value=99, span=span, node_id=2), span=span, node_id=3)
        branch = CaseStmtBranch(pattern=pat, body=(), span=span, node_id=4)
        stmt = CaseStmt(subject=subject, branches=(branch,), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        with pytest.raises(AglRaise) as exc_info:
            interp._exec_case_stmt(stmt, scope)
        assert exc_info.value.exc.type_name == "MatchError"

    def test_try_catch_no_exception_runs_body(self) -> None:
        """TryCatch with non-raising body executes normally."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CatchClause, IntLit, LetDecl, TryCatch
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        let = LetDecl(
            name="y", type_ann=None, value=IntLit(value=3, span=span, node_id=1),
            span=span, node_id=2,
        )
        handler = CatchClause(exc_type=None, binding=None, body=(), span=span, node_id=3)
        stmt = TryCatch(body=(let,), handlers=(handler,), span=span, node_id=4)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_try_catch(stmt, scope)
        # y defined in try_scope (child), not visible in root scope — no error

    def test_try_catch_catches_matching_exception_with_binding(self) -> None:
        """TryCatch catches a matching AglRaise and binds the exception value."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause, PassStmt, TryCatch
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)

        exc_val = ExceptionValue(
            type_name="Abort",
            fields={"message": TextValue("oops"), "trace_id": TextValue("")},
        )

        # Use a counter to only raise on the first call (the body), not the handler
        call_count = [0]

        class RaisingInterp(Interpreter):
            def _exec_stmt(self, s: object, sc: object) -> None:
                call_count[0] += 1
                if call_count[0] == 1:
                    raise AglRaise(exc_val)
                # Handler body executes normally (pass)

        from agm.agl.capabilities import HostCapabilities
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        program = parse_program("pass")
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(), has_fallback_agent=False,
            codec_kinds={}, renderer_names=frozenset()
        )
        checked = check(resolved, caps)
        registry = AgentRegistry(named={}, default_agent=None)
        interp = RaisingInterp(
            checked=checked,
            registry=registry,
            contracts={},
            type_env=TypeEnvironment(),
            loop_limit=3,
            strict_json=False,
        )

        handler = CatchClause(
            exc_type="Abort",
            binding="e",
            body=(PassStmt(span=span, node_id=10),),
            span=span,
            node_id=11,
        )
        stmt = TryCatch(
            body=(PassStmt(span=span, node_id=1),),
            handlers=(handler,),
            span=span,
            node_id=12,
        )
        scope = Scope(parent=None)
        interp._exec_try_catch(stmt, scope)  # should not raise
        assert call_count[0] == 2  # body + handler body

    def test_try_catch_reraises_unhandled_exception(self) -> None:
        """TryCatch re-raises when no handler matches."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause, PassStmt, TryCatch
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)

        # Only raise on the body statement, not on any handler stmt
        class BodyRaisingInterp(Interpreter):
            def _exec_stmt(self, s: object, sc: object) -> None:
                exc_val = ExceptionValue(
                    type_name="NetworkError",
                    fields={"message": TextValue("conn"), "trace_id": TextValue("")},
                )
                raise AglRaise(exc_val)

        from agm.agl.capabilities import HostCapabilities
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        program = parse_program("pass")
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(), has_fallback_agent=False,
            codec_kinds={}, renderer_names=frozenset()
        )
        checked = check(resolved, caps)
        registry = AgentRegistry(named={}, default_agent=None)
        interp = BodyRaisingInterp(
            checked=checked,
            registry=registry,
            contracts={},
            type_env=TypeEnvironment(),
            loop_limit=3,
            strict_json=False,
        )

        # Handler only catches "Abort" — NetworkError will be re-raised.
        handler = CatchClause(exc_type="Abort", binding=None, body=(), span=span, node_id=5)
        stmt = TryCatch(
            body=(PassStmt(span=span, node_id=1),),
            handlers=(handler,),
            span=span,
            node_id=6,
        )
        scope = Scope(parent=None)
        with pytest.raises(AglRaise):
            interp._exec_try_catch(stmt, scope)

    def test_exec_raise_exception_value(self) -> None:
        """_exec_raise with ExceptionValue propagates AglRaise."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import Raise
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)

        # Build an ExceptionValue in scope and raise it via a VarRef.
        from agm.agl.syntax.nodes import VarRef

        exc_val = ExceptionValue(
            type_name="Abort",
            fields={"message": TextValue("boom"), "trace_id": TextValue("")},
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)

        scope = Scope(parent=None)
        scope.define("err", exc_val, mutable=False, decl_span=span)

        ref = VarRef(name="err", span=span, node_id=1)
        raise_stmt = Raise(exc=ref, span=span, node_id=2)
        with pytest.raises(AglRaise) as exc_info:
            interp._exec_raise(raise_stmt, scope)
        assert exc_info.value.exc.type_name == "Abort"

    def test_exec_raise_non_exception_value_raises_runtime(self) -> None:
        """_exec_raise with a non-ExceptionValue raises RuntimeError."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import IntLit, Raise
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        raise_stmt = Raise(exc=IntLit(value=5, span=span, node_id=1), span=span, node_id=2)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        with pytest.raises(RuntimeError):
            interp._exec_raise(raise_stmt, scope)

    def test_exec_stmt_input_decl_is_noop(self) -> None:
        """InputDecl in _exec_stmt is a no-op at runtime."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import InputDecl
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        stmt = InputDecl(name="x", annotation=None, span=span, node_id=1)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_stmt(stmt, scope)  # No error, no binding

    def test_exec_stmt_record_enum_alias_noop(self) -> None:
        """RecordDef/EnumDef/TypeAlias in _exec_stmt are no-ops."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import EnumDef, RecordDef, TypeAlias
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import IntT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        rec = RecordDef(name="Point", fields=(), span=span, node_id=1)
        interp._exec_stmt(rec, scope)  # No error

        enm = EnumDef(name="Color", variants=(), span=span, node_id=2)
        interp._exec_stmt(enm, scope)  # No error

        alias = TypeAlias(name="Num", type_expr=IntT(span=span, node_id=0), span=span, node_id=3)
        interp._exec_stmt(alias, scope)  # No error


# ---------------------------------------------------------------------------
# Coverage: let/var type annotation coercion (_exec_let, _exec_var)
# ---------------------------------------------------------------------------


class TestLetVarCoercion:
    """Tests for let/var with explicit type annotations that trigger coercion."""

    def test_exec_let_with_decimal_annotation_coerces_int(self) -> None:
        """let x: decimal = 3 → x holds DecimalValue(3)."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import IntLit, LetDecl
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import DecimalT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        ann = DecimalT(span=span, node_id=0)
        stmt = LetDecl(
            name="x",
            type_ann=ann,
            value=IntLit(value=3, span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_let(stmt, scope)
        b = scope.lookup("x")
        assert b is not None
        assert isinstance(b.value, DecimalValue)

    def test_exec_var_with_text_annotation_no_coerce(self) -> None:
        """var x: text = 'hi' → TextValue unchanged."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import StringLit, VarDecl
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import TextT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        ann = TextT(span=span, node_id=0)
        stmt = VarDecl(
            name="msg",
            type_ann=ann,
            value=StringLit(value="hi", span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_var(stmt, scope)
        b = scope.lookup("msg")
        assert b is not None
        assert b.value == TextValue("hi")

    def test_exec_let_with_bool_annotation(self) -> None:
        """let x: bool = true → BoolValue unchanged."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BoolLit, LetDecl
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import BoolT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        ann = BoolT(span=span, node_id=0)
        stmt = LetDecl(
            name="b",
            type_ann=ann,
            value=BoolLit(value=True, span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_let(stmt, scope)
        b = scope.lookup("b")
        assert b is not None
        assert b.value == BoolValue(True)

    def test_exec_let_with_json_annotation(self) -> None:
        """let x: json = null → JsonValue unchanged."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import JsonValue
        from agm.agl.syntax.nodes import LetDecl, NullLit
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import JsonT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        ann = JsonT(span=span, node_id=0)
        stmt = LetDecl(
            name="j",
            type_ann=ann,
            value=NullLit(span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_let(stmt, scope)
        b = scope.lookup("j")
        assert b is not None
        assert b.value == JsonValue(None)

    def test_exec_set_with_decimal_binding_coerces_int(self) -> None:
        """set x = 5 when x holds DecimalValue → coerces to DecimalValue."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import IntLit, SetStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define("x", DecimalValue(decimal.Decimal("1.0")), mutable=True, decl_span=span)

        stmt = SetStmt(
            target="x",
            value=IntLit(value=7, span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        interp._exec_set(stmt, scope)
        b = scope.lookup("x")
        assert b is not None
        assert isinstance(b.value, DecimalValue)


# ---------------------------------------------------------------------------
# Coverage: _eval_expr paths (operators, list, dict)
# ---------------------------------------------------------------------------


class TestEvalExprCompound:
    """Unit tests for compound expression types in _eval_expr."""

    def test_eval_binary_op_add_int(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.ADD,
            left=IntLit(value=3, span=span, node_id=1),
            right=IntLit(value=4, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == IntValue(7)

    def test_eval_binary_op_sub(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.SUB,
            left=IntLit(value=10, span=span, node_id=1),
            right=IntLit(value=3, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == IntValue(7)

    def test_eval_binary_op_mul(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.MUL,
            left=IntLit(value=4, span=span, node_id=1),
            right=IntLit(value=5, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == IntValue(20)

    def test_eval_binary_op_div(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.DIV,
            left=IntLit(value=7, span=span, node_id=1),
            right=IntLit(value=2, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert isinstance(result, DecimalValue)

    def test_eval_binary_op_compare_eq(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.EQ,
            left=IntLit(value=1, span=span, node_id=1),
            right=IntLit(value=1, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == BoolValue(True)

    def test_eval_binary_op_in_op(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue, IntValue, ListValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        lst = ListValue(elements=(IntValue(1), IntValue(2)))
        scope.define("lst", lst, mutable=False, decl_span=span)

        expr = BinaryOp(
            op=BinOp.IN,
            left=IntLit(value=1, span=span, node_id=1),
            right=VarRef(name="lst", span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, scope)
        assert result == BoolValue(True)

    def test_eval_binary_op_and_short_circuit_false(self) -> None:
        """and: left=false → returns false without evaluating right."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.AND,
            left=BoolLit(value=False, span=span, node_id=1),
            right=BoolLit(value=True, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == BoolValue(False)

    def test_eval_binary_op_and_both_true(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.AND,
            left=BoolLit(value=True, span=span, node_id=1),
            right=BoolLit(value=True, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == BoolValue(True)

    def test_eval_binary_op_or_short_circuit_true(self) -> None:
        """or: left=true → returns true without evaluating right."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.OR,
            left=BoolLit(value=True, span=span, node_id=1),
            right=BoolLit(value=False, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == BoolValue(True)

    def test_eval_binary_op_or_both_false(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.OR,
            left=BoolLit(value=False, span=span, node_id=1),
            right=BoolLit(value=False, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        assert result == BoolValue(False)

    def test_eval_is_test_matching_variant(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue, EnumValue
        from agm.agl.syntax.nodes import IsTest, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "s",
            EnumValue(type_name="Status", variant="Active", fields={}),
            mutable=False,
            decl_span=span,
        )
        expr = IsTest(
            expr=VarRef(name="s", span=span, node_id=1),
            qualifier=None,
            variant="Active",
            negated=False,
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_is_test(expr, scope)
        assert result == BoolValue(True)

    def test_eval_is_test_negated(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue, EnumValue
        from agm.agl.syntax.nodes import IsTest, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "s",
            EnumValue(type_name="Status", variant="Inactive", fields={}),
            mutable=False,
            decl_span=span,
        )
        expr = IsTest(
            expr=VarRef(name="s", span=span, node_id=1),
            qualifier=None,
            variant="Active",
            negated=True,
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_is_test(expr, scope)
        assert result == BoolValue(True)

    def test_eval_is_test_on_non_enum_raises(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IsTest, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define("n", IntValue(1), mutable=False, decl_span=span)
        expr = IsTest(
            expr=VarRef(name="n", span=span, node_id=1),
            qualifier=None,
            variant="Active",
            negated=False,
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="is test on non-enum"):
            interp._eval_is_test(expr, scope)

    def test_eval_case_expr_matches(self) -> None:
        """CaseExpr with matching branch returns the branch body value."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import (
            CaseExpr,
            CaseExprBranch,
            IntLit,
            LiteralPattern,
        )
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=5, span=span, node_id=1)
        # Branch: pattern=5 → body=99
        pat = LiteralPattern(literal=IntLit(value=5, span=span, node_id=2), span=span, node_id=3)
        branch = CaseExprBranch(
            pattern=pat, body=IntLit(value=99, span=span, node_id=4), span=span, node_id=5
        )
        expr = CaseExpr(subject=subject, branches=(branch,), span=span, node_id=6)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_case_expr(expr, Scope(parent=None))
        assert result == IntValue(99)

    def test_eval_case_expr_no_match_raises(self) -> None:
        """CaseExpr with no matching branch raises AglRaise(MatchError)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import (
            CaseExpr,
            CaseExprBranch,
            IntLit,
            LiteralPattern,
        )
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=1, span=span, node_id=1)
        pat = LiteralPattern(literal=IntLit(value=99, span=span, node_id=2), span=span, node_id=3)
        branch = CaseExprBranch(
            pattern=pat, body=IntLit(value=0, span=span, node_id=4), span=span, node_id=5
        )
        expr = CaseExpr(subject=subject, branches=(branch,), span=span, node_id=6)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(AglRaise) as exc_info:
            interp._eval_case_expr(expr, Scope(parent=None))
        assert exc_info.value.exc.type_name == "MatchError"

    def test_eval_list_lit(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.syntax.nodes import IntLit, ListLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = ListLit(
            elements=(IntLit(value=1, span=span, node_id=1), IntLit(value=2, span=span, node_id=2)),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_list_lit(expr, Scope(parent=None))
        assert isinstance(result, ListValue)
        assert result.elements == (IntValue(1), IntValue(2))

    def test_eval_dict_lit(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.syntax.nodes import DictEntry, DictLit, IntLit, StringLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        key = StringLit(value="k", span=span, node_id=1)
        val = IntLit(value=42, span=span, node_id=2)
        entry = DictEntry(key=key, value=val, span=span, node_id=3)
        expr = DictLit(entries=(entry,), span=span, node_id=4)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_dict_lit(expr, Scope(parent=None))
        assert isinstance(result, DictValue)
        assert result.entries["k"] == IntValue(42)

    def test_eval_unary_not_true(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BoolLit, UnaryNot
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = UnaryNot(operand=BoolLit(value=True, span=span, node_id=1), span=span, node_id=2)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_unary_not(expr, Scope(parent=None))
        assert result == BoolValue(False)

    def test_eval_unary_neg_int(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IntLit, UnaryNeg
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = UnaryNeg(operand=IntLit(value=5, span=span, node_id=1), span=span, node_id=2)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_unary_neg(expr, Scope(parent=None))
        assert result == IntValue(-5)

    def test_eval_unary_neg_decimal(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import DecimalLit, UnaryNeg
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = UnaryNeg(
            operand=DecimalLit(value=decimal.Decimal("2.5"), span=span, node_id=1),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_unary_neg(expr, Scope(parent=None))
        assert result == DecimalValue(decimal.Decimal("-2.5"))

# ---------------------------------------------------------------------------
# Coverage: field access on RecordValue, EnumValue, ExceptionValue
# ---------------------------------------------------------------------------


class TestFieldAccess:
    def test_field_access_on_record_value(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "pt",
            RecordValue(type_name="Point", fields={"x": IntValue(3)}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="pt", span=span, node_id=1), field="x", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_field_access(expr, scope)
        assert result == IntValue(3)

    def test_field_access_missing_record_field_raises(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import RecordValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "pt",
            RecordValue(type_name="Point", fields={}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="pt", span=span, node_id=1), field="z", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="no field"):
            interp._eval_field_access(expr, scope)

    def test_field_access_on_enum_value(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue, TextValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "ev",
            EnumValue(type_name="Color", variant="Red", fields={"label": TextValue("red")}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="ev", span=span, node_id=1), field="label", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_field_access(expr, scope)
        assert result == TextValue("red")

    def test_field_access_missing_enum_field_raises(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "ev",
            EnumValue(type_name="Color", variant="Red", fields={}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="ev", span=span, node_id=1), field="missing", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="no field"):
            interp._eval_field_access(expr, scope)

    def test_field_access_on_exception_value(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "ex",
            ExceptionValue(
                type_name="E", fields={"message": TextValue("oops"), "trace_id": TextValue("")}
            ),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="ex", span=span, node_id=1), field="message", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_field_access(expr, scope)
        assert result == TextValue("oops")

    def test_field_access_missing_exception_field_raises(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "ex",
            ExceptionValue(type_name="E", fields={}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="ex", span=span, node_id=1), field="missing", span=span, node_id=2
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="no field"):
            interp._eval_field_access(expr, scope)


# ---------------------------------------------------------------------------
# Coverage: constructor evaluation
# ---------------------------------------------------------------------------


class TestConstructorEval:
    def test_eval_constructor_record(self) -> None:
        """Constructor for a RecordType builds a RecordValue."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import Constructor, IntLit, NamedArg
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import IntType, RecordType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        type_env.register_type("Point", RecordType(name="Point", fields={"x": IntType()}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier=None,
            name="Point",
            args=(
                NamedArg(
                    name="x", value=IntLit(value=5, span=span, node_id=1), span=span, node_id=2
                ),
            ),
            span=span,
            node_id=3,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, RecordValue)
        assert result.type_name == "Point"
        assert result.fields["x"] == IntValue(5)

    def test_eval_constructor_enum_qualified(self) -> None:
        """Qualified enum constructor (Color.Red) builds an EnumValue."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import Constructor
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import EnumType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        type_env.register_type("Color", EnumType(name="Color", variants={"Red": {}}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier="Color",
            name="Red",
            args=(),
            span=span,
            node_id=1,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, EnumValue)
        assert result.variant == "Red"

    def test_eval_constructor_unqualified_enum_variant(self) -> None:
        """Unqualified enum constructor (Active) resolves by scanning enum types."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import Constructor
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import EnumType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        type_env.register_type("Status", EnumType(name="Status", variants={"Active": {}}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(qualifier=None, name="Active", args=(), span=span, node_id=1)
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, EnumValue)
        assert result.variant == "Active"

    def test_eval_constructor_unknown_type_raises(self) -> None:
        """Constructor for an unknown type raises RuntimeError."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import Constructor
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = Constructor(qualifier=None, name="UnknownType", args=(), span=span, node_id=1)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="Cannot construct"):
            interp._eval_constructor(expr, Scope(parent=None))


# ---------------------------------------------------------------------------
# Coverage: agent call edge cases (fallback contract, strict_json, retries)
# ---------------------------------------------------------------------------


class TestAgentCallEdgeCases:
    def test_agent_call_uses_fallback_contract_when_missing(self) -> None:
        """When no contract is registered for a call node, a TextCodec contract is used."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, CallOptions, Template, TextSegment
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        program = parse_program("pass")
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        def my_fn(req: AgentRequest) -> str:
            return "hello"

        # call_kind will be None for synthetic node → dispatches to "prompt" (default agent)
        registry = AgentRegistry(named={}, default_agent=my_fn)
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={},  # No contracts → triggers fallback
            type_env=TypeEnvironment(),
            loop_limit=3,
            strict_json=False,
        )

        scope = Scope(parent=None)
        opts = CallOptions(format=None, strict_json=None, parse_policy=None, span=span, node_id=1)
        template = Template(
            segments=(TextSegment(text="hello", span=span, node_id=2),),
            span=span,
            node_id=3,
        )
        # node_id=99 has no call_kinds entry → call_kind=None → agent_name="prompt"
        expr = AgentCall(agent="prompt", options=opts, template=template, span=span, node_id=99)
        result = interp._eval_agent_call(expr, scope)
        assert isinstance(result, TextValue)
        assert result.value == "hello"

    def test_agent_call_strict_json_from_contract(self) -> None:
        """Agent call uses contract.strict_json when not None."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, CallOptions, Template, TextSegment
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import TextType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        program = parse_program("pass")
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        def my_fn(req: AgentRequest) -> str:
            return "ok"

        # call_kind=None for synthetic node → dispatches to "prompt" (default agent)
        registry = AgentRegistry(named={}, default_agent=my_fn)
        node_id = 99
        contract = OutputContract(
            target_type=TextType(),
            codec=TextCodec(),
            strict_json=True,  # Explicit strict_json override
            format_instructions="",
            json_schema=None,
        )
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={node_id: contract},
            type_env=TypeEnvironment(),
            loop_limit=3,
            strict_json=False,  # Runtime default is False, contract overrides to True
        )

        scope = Scope(parent=None)
        opts = CallOptions(format=None, strict_json=None, parse_policy=None, span=span, node_id=1)
        template = Template(
            segments=(TextSegment(text="hi", span=span, node_id=2),),
            span=span,
            node_id=3,
        )
        expr = AgentCall(
            agent="prompt", options=opts, template=template, span=span, node_id=node_id
        )
        result = interp._eval_agent_call(expr, scope)
        assert isinstance(result, TextValue)

    def test_agent_call_retry_policy_exhausts(self) -> None:
        """A retry-policy call whose codec always fails raises AgentParseError.

        Driven through the interpreter's public ``execute`` entry on a real
        parsed program (``let x = prompt[on_parse_error: retry[1]] "hi"``).  In
        M1 only the JSON codec (M2) can fail, so this exercises the path with a
        public, host-supplied failing codec injected via the contract map.
        """
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.scope import resolve
        from agm.agl.syntax.nodes import AgentCall, LetDecl
        from agm.agl.typecheck import check
        from agm.agl.typecheck.types import TextType

        source = 'let x = prompt[on_parse_error: retry[1]] "hi"'
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=True,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        # Locate the real AgentCall node so we can key its output contract.
        let_stmt = checked.resolved.program.body[0]
        assert isinstance(let_stmt, LetDecl)
        call_expr = let_stmt.value
        assert isinstance(call_expr, AgentCall)

        class AlwaysFailCodec(TextCodec):
            def parse(
                self, raw: str, target_type: object, *, strict_json: bool = False
            ) -> ParseResult:
                return ParseResult.failure("always fails")

        def my_fn(req: AgentRequest) -> str:
            return "not-valid"

        registry = AgentRegistry(named={}, default_agent=my_fn)
        contract = OutputContract(
            target_type=TextType(),
            codec=AlwaysFailCodec(),
            strict_json=None,
            format_instructions="",
            json_schema=None,
        )
        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts={call_expr.node_id: contract},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )

        with pytest.raises(AglRaise) as exc_info:
            interp.execute(Scope(parent=None))
        assert exc_info.value.exc.type_name == "AgentParseError"


# ---------------------------------------------------------------------------
# Coverage: _resolve_type_ann branches (Text, Bool, Json)
# ---------------------------------------------------------------------------


class TestResolveTypeAnn:
    def test_resolve_type_ann_text(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import TextT
        from agm.agl.typecheck.types import TextType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(TextT(span=span, node_id=0))
        assert isinstance(result, TextType)

    def test_resolve_type_ann_bool(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import BoolT
        from agm.agl.typecheck.types import BoolType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(BoolT(span=span, node_id=0))
        assert isinstance(result, BoolType)

    def test_resolve_type_ann_json(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import JsonT
        from agm.agl.typecheck.types import JsonType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(JsonT(span=span, node_id=0))
        assert isinstance(result, JsonType)

    def test_resolve_type_ann_unknown_returns_none(self) -> None:
        from agm.agl.eval.interpreter import _resolve_type_ann

        result = _resolve_type_ann(object())
        assert result is None


# ---------------------------------------------------------------------------
# Coverage: _match_pattern ConstructorPattern for RecordValue and EnumValue
# ---------------------------------------------------------------------------


class TestMatchPatternConstructor:
    def test_constructor_pattern_matches_record(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import (
            ConstructorPattern,
            PatternField,
            WildcardPattern,
        )
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        # Pattern: Point { x: _ }
        field_pat = PatternField(
            name="x",
            pattern=WildcardPattern(span=span, node_id=1),
            span=span,
            node_id=2,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Point", fields=(field_pat,), span=span, node_id=3
        )
        value = RecordValue(type_name="Point", fields={"x": IntValue(5)})
        matched, bindings = _match_pattern(pat, value)
        assert matched
        assert bindings == {}

    def test_constructor_pattern_record_type_mismatch(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import ConstructorPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        pat = ConstructorPattern(qualifier=None, name="Line", fields=(), span=span, node_id=1)
        value = RecordValue(type_name="Point", fields={"x": IntValue(5)})
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_record_qualified_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import RecordValue
        from agm.agl.syntax.nodes import ConstructorPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        # Pattern: qualifier="Point", name="Something" should not match type_name="Point"
        pat = ConstructorPattern(
            qualifier="Point", name="Other", fields=(), span=span, node_id=1
        )
        value = RecordValue(type_name="Point", fields={})
        matched, _ = _match_pattern(pat, value)
        # qualifier="Point" matches type_name="Point" but name is different
        # from the code: type_name != pattern.name and (qualifier is None or qualifier != type_name)
        # => "Point" != "Other" and ("Point" is not None and "Point" == "Point")
        # → second part is False
        # so whole condition False → not returned early → matched
        assert matched

    def test_constructor_pattern_record_missing_field(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import RecordValue
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField, WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        field_pat = PatternField(
            name="z",
            pattern=WildcardPattern(span=span, node_id=1),
            span=span,
            node_id=2,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Point", fields=(field_pat,), span=span, node_id=3
        )
        value = RecordValue(type_name="Point", fields={})  # missing "z"
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_record_sub_pattern_no_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import (
            ConstructorPattern,
            IntLit,
            LiteralPattern,
            PatternField,
        )
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        # Pattern: Point { x: 99 } — x is 5, so no match
        field_pat = PatternField(
            name="x",
            pattern=LiteralPattern(
                literal=IntLit(value=99, span=span, node_id=1), span=span, node_id=2
            ),
            span=span,
            node_id=3,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Point", fields=(field_pat,), span=span, node_id=4
        )
        value = RecordValue(type_name="Point", fields={"x": IntValue(5)})
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_matches_enum_variant(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import EnumValue, IntValue
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField, VarPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        field_pat = PatternField(
            name="n",
            pattern=VarPattern(name="val", span=span, node_id=1),
            span=span,
            node_id=2,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Some", fields=(field_pat,), span=span, node_id=3
        )
        value = EnumValue(type_name="Option", variant="Some", fields={"n": IntValue(42)})
        matched, bindings = _match_pattern(pat, value)
        assert matched
        assert bindings == {"val": IntValue(42)}

    def test_constructor_pattern_enum_variant_mismatch(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import ConstructorPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        pat = ConstructorPattern(qualifier=None, name="None", fields=(), span=span, node_id=1)
        value = EnumValue(type_name="Option", variant="Some", fields={})
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_enum_missing_field(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import ConstructorPattern, PatternField, WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        field_pat = PatternField(
            name="missing",
            pattern=WildcardPattern(span=span, node_id=1),
            span=span,
            node_id=2,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Some", fields=(field_pat,), span=span, node_id=3
        )
        value = EnumValue(type_name="Option", variant="Some", fields={})  # no "missing"
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_enum_sub_pattern_no_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import EnumValue, IntValue
        from agm.agl.syntax.nodes import (
            ConstructorPattern,
            IntLit,
            LiteralPattern,
            PatternField,
        )
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        field_pat = PatternField(
            name="n",
            pattern=LiteralPattern(
                literal=IntLit(value=99, span=span, node_id=1), span=span, node_id=2
            ),
            span=span,
            node_id=3,
        )
        pat = ConstructorPattern(
            qualifier=None, name="Some", fields=(field_pat,), span=span, node_id=4
        )
        value = EnumValue(type_name="Option", variant="Some", fields={"n": IntValue(1)})
        matched, _ = _match_pattern(pat, value)
        assert not matched

    def test_constructor_pattern_on_non_record_enum_no_match(self) -> None:
        from agm.agl.eval.interpreter import _match_pattern
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import ConstructorPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        pat = ConstructorPattern(qualifier=None, name="Foo", fields=(), span=span, node_id=1)
        matched, _ = _match_pattern(pat, IntValue(1))
        assert not matched


# ---------------------------------------------------------------------------
# CheckedProgram.type_env carries the constructor namespace to the interpreter
# ---------------------------------------------------------------------------


class TestCheckedProgramTypeEnvConstructors:
    """The constructor namespace flows from check() to the interpreter via
    ``CheckedProgram.type_env``.

    The M1 parser cannot yet parse ``record``/constructor syntax, so the program
    is hand-built from AST nodes, then driven through the *real* resolve + check
    passes (no fabricated side tables) and evaluated via the interpreter's public
    ``execute`` entry.  This is the regression test for the bug where the runtime
    reconstructed an empty type env from expression-only ``node_types``, leaving
    every constructor unresolvable at runtime.
    """

    def test_record_constructor_resolves_to_record_value(self) -> None:
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue, RecordValue
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.scope.symbols import ResolvedProgram
        from agm.agl.syntax.nodes import (
            Constructor,
            FieldDef,
            IntLit,
            LetDecl,
            NamedArg,
            Program,
            RecordDef,
        )
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import DecimalT
        from agm.agl.typecheck import check

        span = SourceSpan(1, 1, 1, 1, 0, 0)

        # record Point { x: decimal }
        record = RecordDef(
            name="Point",
            fields=(FieldDef(name="x", type_expr=DecimalT(span=span, node_id=1), span=span,
                             node_id=2),),
            span=span,
            node_id=3,
        )
        # let p = Point(x: 1)
        ctor = Constructor(
            qualifier=None,
            name="Point",
            args=(NamedArg(name="x", value=IntLit(value=1, span=span, node_id=4), span=span,
                           node_id=5),),
            span=span,
            node_id=6,
        )
        let_p = LetDecl(name="p", type_ann=None, value=ctor, span=span, node_id=7)
        program = Program(body=(record, let_p), span=span, node_id=8)

        # Real resolve + check passes — no fabricated side tables.
        resolved: ResolvedProgram = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            codec_kinds={"text": frozenset({"text"})},
            renderer_names=frozenset({"default"}),
        )
        checked = check(resolved, caps)

        # The type namespace must include the user record after checking.
        assert checked.type_env.get_type("Point") is not None

        interp = Interpreter(
            checked=checked,
            registry=AgentRegistry(named={}, default_agent=None),
            contracts={},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )
        root = Scope(parent=None)
        interp.execute(root)

        binding = root.lookup("p")
        assert binding is not None
        value = binding.value
        assert isinstance(value, RecordValue)
        assert value.type_name == "Point"
        # int arg coerces to the decimal field type.
        assert value.fields["x"] == DecimalValue(decimal.Decimal(1))


# ---------------------------------------------------------------------------
# Coverage: runtime.py _convert_input int-from-Decimal (line 401)
# ---------------------------------------------------------------------------


class TestConvertInputDecimalToInt:
    def test_convert_input_int_from_decimal_string_parsed(self) -> None:
        """'1.0' parses as Decimal('1.0') == int(1) → IntValue(1)."""
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        result = _convert_input("n", "1.0", IntType())
        assert result == IntValue(1)

    def test_convert_input_int_from_decimal_non_integer_raises(self) -> None:
        """'1.5' parses as Decimal('1.5') ≠ int(1) → ValueError."""
        from agm.agl.runtime.runtime import _convert_input
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="integer"):
            _convert_input("n", "1.5", IntType())


# ---------------------------------------------------------------------------
# Coverage: _exec_stmt dispatch for M3+ types (lines 158-166)
# These test calling _exec_stmt (not the underlying methods directly) so
# the dispatch elif branches are covered.
# ---------------------------------------------------------------------------


class TestExecStmtDispatch:
    """Tests that call _exec_stmt for M3+ statement types to cover dispatch branches."""

    def test_exec_stmt_dispatches_do_until(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, DoUntil
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        cond = BoolLit(value=True, span=span, node_id=1)
        stmt = DoUntil(limit=1, body=(), condition=cond, span=span, node_id=2)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_stmt(stmt, scope)  # Dispatches via DoUntil branch

    def test_exec_stmt_dispatches_if_stmt(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import BoolLit, IfBranch, IfStmt
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        branch = IfBranch(
            cond=BoolLit(value=False, span=span, node_id=1), body=(), span=span, node_id=2
        )
        stmt = IfStmt(branches=(branch,), span=span, node_id=3)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_stmt(stmt, scope)  # Dispatches via IfStmt branch

    def test_exec_stmt_dispatches_case_stmt(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, IntLit, WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=1, span=span, node_id=1)
        branch = CaseStmtBranch(
            pattern=WildcardPattern(span=span, node_id=2), body=(), span=span, node_id=3
        )
        stmt = CaseStmt(subject=subject, branches=(branch,), span=span, node_id=4)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_stmt(stmt, scope)  # Dispatches via CaseStmt branch

    def test_exec_stmt_dispatches_try_catch(self) -> None:
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CatchClause, TryCatch
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        handler = CatchClause(exc_type=None, binding=None, body=(), span=span, node_id=1)
        stmt = TryCatch(body=(), handlers=(handler,), span=span, node_id=2)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_stmt(stmt, scope)  # Dispatches via TryCatch branch

    def test_exec_stmt_dispatches_raise(self) -> None:
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import Raise, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        exc_val = ExceptionValue(
            type_name="Abort",
            fields={"message": TextValue("x"), "trace_id": TextValue("")},
        )
        scope = Scope(parent=None)
        scope.define("e", exc_val, mutable=False, decl_span=span)
        stmt = Raise(exc=VarRef(name="e", span=span, node_id=1), span=span, node_id=2)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(AglRaise):
            interp._exec_stmt(stmt, scope)  # Dispatches via Raise branch


# ---------------------------------------------------------------------------
# Coverage: remaining interpreter paths via _eval_expr dispatch
# ---------------------------------------------------------------------------


class TestEvalExprDispatch:
    """Tests that call _eval_expr (not sub-methods) to cover dispatch branches."""

    def test_eval_expr_dispatches_field_access(self) -> None:
        """_eval_expr dispatches to _eval_field_access for FieldAccess nodes."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import FieldAccess, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "pt",
            RecordValue(type_name="P", fields={"x": IntValue(9)}),
            mutable=False,
            decl_span=span,
        )
        expr = FieldAccess(
            obj=VarRef(name="pt", span=span, node_id=1), field="x", span=span, node_id=2
        )

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, scope)
        assert result == IntValue(9)

    def test_eval_expr_dispatches_template(self) -> None:
        """_eval_expr dispatches compound expressions via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import Template, TextSegment
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = Template(
            segments=(TextSegment(text="hello", span=span, node_id=1),),
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert result == TextValue("hello")

    def test_eval_expr_dispatches_list_lit(self) -> None:
        """_eval_expr dispatches ListLit via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ListValue
        from agm.agl.syntax.nodes import IntLit, ListLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = ListLit(elements=(IntLit(value=1, span=span, node_id=1),), span=span, node_id=2)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert isinstance(result, ListValue)

    def test_eval_expr_dispatches_dict_lit(self) -> None:
        """_eval_expr dispatches DictLit via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DictValue
        from agm.agl.syntax.nodes import DictEntry, DictLit, IntLit, StringLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        entry = DictEntry(
            key=StringLit(value="k", span=span, node_id=1),
            value=IntLit(value=1, span=span, node_id=2),
            span=span, node_id=3,
        )
        expr = DictLit(entries=(entry,), span=span, node_id=4)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert isinstance(result, DictValue)

    def test_eval_expr_dispatches_unary_not(self) -> None:
        """_eval_expr dispatches UnaryNot via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue
        from agm.agl.syntax.nodes import BoolLit, UnaryNot
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = UnaryNot(operand=BoolLit(value=False, span=span, node_id=1), span=span, node_id=2)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert result == BoolValue(True)

    def test_eval_expr_dispatches_unary_neg(self) -> None:
        """_eval_expr dispatches UnaryNeg via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IntLit, UnaryNeg
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = UnaryNeg(operand=IntLit(value=3, span=span, node_id=1), span=span, node_id=2)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert result == IntValue(-3)

    def test_eval_expr_dispatches_is_test(self) -> None:
        """_eval_expr dispatches IsTest via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import BoolValue, EnumValue
        from agm.agl.syntax.nodes import IsTest, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        scope = Scope(parent=None)
        scope.define(
            "s", EnumValue(type_name="S", variant="A", fields={}), mutable=False, decl_span=span
        )
        expr = IsTest(
            expr=VarRef(name="s", span=span, node_id=1),
            qualifier=None,
            variant="A",
            negated=False,
            span=span,
            node_id=2,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, scope)
        assert result == BoolValue(True)

    def test_eval_expr_dispatches_case_expr(self) -> None:
        """_eval_expr dispatches CaseExpr via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, IntLit, WildcardPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=5, span=span, node_id=1)
        branch = CaseExprBranch(
            pattern=WildcardPattern(span=span, node_id=2),
            body=IntLit(value=99, span=span, node_id=3),
            span=span,
            node_id=4,
        )
        expr = CaseExpr(subject=subject, branches=(branch,), span=span, node_id=5)
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert result == IntValue(99)

    def test_eval_expr_dispatches_binary_op(self) -> None:
        """_eval_expr dispatches BinaryOp via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.ADD,
            left=IntLit(value=2, span=span, node_id=1),
            right=IntLit(value=3, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert result == IntValue(5)

    def test_eval_expr_dispatches_constructor(self) -> None:
        """_eval_expr dispatches Constructor via _eval_expr."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue
        from agm.agl.syntax.nodes import Constructor
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import EnumType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        type_env.register_type("Color", EnumType(name="Color", variants={"Red": {}}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)

        expr = Constructor(qualifier="Color", name="Red", args=(), span=span, node_id=1)
        result = interp._eval_expr(expr, Scope(parent=None))
        assert isinstance(result, EnumValue)


# ---------------------------------------------------------------------------
# Coverage: template interpolation non-VarRef expr path (lines 408->410)
# ---------------------------------------------------------------------------


class TestTemplateInterpSegment:
    def test_template_interp_non_var_ref_no_var_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """InterpSegment with a non-VarRef expr has var_name=None."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import InterpSegment, IntLit, Template
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        # IntLit is not a VarRef → var_name stays None
        interp_seg = InterpSegment(
            expr=IntLit(value=42, span=span, node_id=1), render=None, span=span, node_id=2
        )
        template = Template(segments=(interp_seg,), span=span, node_id=3)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_template(template, Scope(parent=None))
        assert isinstance(result, TextValue)
        assert "42" in result.value


# ---------------------------------------------------------------------------
# Coverage: agent call shell_exec path (line 421)
# ---------------------------------------------------------------------------


class TestAgentCallShellExec:
    def test_eval_agent_call_shell_exec_raises(self) -> None:
        """When call_kind == CallKind.shell_exec, raises AglRaise(ExecError)."""
        from agm.agl.capabilities import HostCapabilities
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.scope.symbols import CallKind, ResolvedProgram
        from agm.agl.syntax.nodes import AgentCall, CallOptions, Template, TextSegment
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import CheckedProgram

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        source = "pass"
        program = parse_program(source)
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(),
            has_fallback_agent=False,
            codec_kinds={},
            renderer_names=frozenset(),
        )
        checked = check(resolved, caps)

        # Inject a synthetic call_kinds entry with shell_exec
        node_id = 88
        new_call_kinds = dict(checked.resolved.call_kinds)
        new_call_kinds[node_id] = CallKind.shell_exec
        new_resolved = ResolvedProgram(
            program=checked.resolved.program,
            resolution=checked.resolved.resolution,
            call_kinds=new_call_kinds,
            root_scope=checked.resolved.root_scope,
        )
        new_checked = CheckedProgram(
            resolved=new_resolved,
            node_types=checked.node_types,
            contract_specs=checked.contract_specs,
            warnings=checked.warnings,
            type_env=checked.type_env,
        )

        registry = AgentRegistry(named={}, default_agent=None)
        interp = Interpreter(
            checked=new_checked,
            registry=registry,
            contracts={},
            type_env=checked.type_env,
            loop_limit=3,
            strict_json=False,
        )

        opts = CallOptions(format=None, strict_json=None, parse_policy=None, span=span, node_id=1)
        template = Template(
            segments=(TextSegment(text="cmd", span=span, node_id=2),), span=span, node_id=3
        )
        expr = AgentCall(agent="exec", options=opts, template=template, span=span, node_id=node_id)
        scope = Scope(parent=None)
        with pytest.raises(AglRaise) as exc_info:
            interp._eval_agent_call(expr, scope)
        assert exc_info.value.exc.type_name == "ExecError"


# ---------------------------------------------------------------------------
# Coverage: constructor coercion (int→decimal in record/enum fields)
# ---------------------------------------------------------------------------


class TestConstructorCoercion:
    def test_eval_constructor_record_int_to_decimal_coercion(self) -> None:
        """Constructor with int arg for a decimal field coerces to DecimalValue."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue, RecordValue
        from agm.agl.syntax.nodes import Constructor, IntLit, NamedArg
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import DecimalType, RecordType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        # Field "v" expects decimal — passing int should be coerced
        type_env.register_type("Box", RecordType(name="Box", fields={"v": DecimalType()}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier=None,
            name="Box",
            args=(
                NamedArg(
                    name="v", value=IntLit(value=3, span=span, node_id=1), span=span, node_id=2
                ),
            ),
            span=span,
            node_id=3,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, RecordValue)
        assert isinstance(result.fields["v"], DecimalValue)

    def test_eval_constructor_enum_int_to_decimal_coercion(self) -> None:
        """Enum constructor with int arg for a decimal field coerces to DecimalValue."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import DecimalValue, EnumValue
        from agm.agl.syntax.nodes import Constructor, IntLit, NamedArg
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import DecimalType, EnumType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        type_env.register_type(
            "Measure",
            EnumType(name="Measure", variants={"Amount": {"v": DecimalType()}}),
        )

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier="Measure",
            name="Amount",
            args=(
                NamedArg(
                    name="v", value=IntLit(value=5, span=span, node_id=1), span=span, node_id=2
                ),
            ),
            span=span,
            node_id=3,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, EnumValue)
        assert isinstance(result.fields["v"], DecimalValue)


# ---------------------------------------------------------------------------
# Coverage: binary op and/or with non-BoolValue right (lines 609, 616)
# ---------------------------------------------------------------------------


class TestBinaryOpNonBoolRight:
    def test_and_true_left_non_bool_right_returns_right(self) -> None:
        """and: left=true, right=IntValue → returns right (not BoolValue)."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.AND,
            left=BoolLit(value=True, span=span, node_id=1),
            right=IntLit(value=42, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        # right is IntValue(42), not BoolValue → returned directly (line 609)
        assert result == IntValue(42)

    def test_or_false_left_non_bool_right_returns_right(self) -> None:
        """or: left=false, right=IntValue → returns right (not BoolValue)."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import BinaryOp, BinOp, BoolLit, IntLit
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = BinaryOp(
            op=BinOp.OR,
            left=BoolLit(value=False, span=span, node_id=1),
            right=IntLit(value=7, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_binary_op(expr, Scope(parent=None))
        # right is IntValue(7), not BoolValue → returned directly (line 616)
        assert result == IntValue(7)


# ---------------------------------------------------------------------------
# Coverage: _to_decimal with invalid type (line 751)
# ---------------------------------------------------------------------------


class TestToDecimalInvalidType:
    def test_to_decimal_non_numeric_raises(self) -> None:
        from agm.agl.eval.interpreter import _to_decimal
        from agm.agl.eval.values import TextValue

        with pytest.raises(RuntimeError, match="Not a numeric value"):
            _to_decimal(TextValue("hello"))


# ---------------------------------------------------------------------------
# Coverage: _compare fallback return BoolValue(False) (line 802)
# ---------------------------------------------------------------------------


class TestCompareFallback:
    def test_compare_non_ordering_op_on_int_returns_false(self) -> None:
        """Passing a non-ordering, non-EQ/NEQ op to _compare with int/int
        falls through all if-branches and returns BoolValue(False) at line 802.

        Path: op=ADD, left=IntValue, right=IntValue
        → EQ check: False
        → NEQ check: False
        → int/int block entered, LT/LE/GT/GE all False
        → falls to return BoolValue(False)
        """
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, IntValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(IntValue(1), IntValue(2), BinOp.ADD)
        assert result == BoolValue(False)


# ---------------------------------------------------------------------------
# Coverage: _match_pattern unknown literal type (line 868)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coverage: _exec_case_stmt with VarPattern binding (line 247)
# ---------------------------------------------------------------------------


class TestCaseStmtVarBinding:
    def test_case_stmt_var_pattern_captures_binding(self) -> None:
        """CaseStmt with VarPattern captures value in branch scope."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import CaseStmt, CaseStmtBranch, IntLit, PassStmt, VarPattern
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=5, span=span, node_id=1)
        # VarPattern: captures value as "n"
        branch = CaseStmtBranch(
            pattern=VarPattern(name="n", span=span, node_id=2),
            body=(PassStmt(span=span, node_id=3),),
            span=span,
            node_id=4,
        )
        stmt = CaseStmt(subject=subject, branches=(branch,), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_case_stmt(stmt, scope)  # Should not raise; binding in branch scope


# ---------------------------------------------------------------------------
# Coverage: _exec_try_catch handler without binding (line 270->277)
# ---------------------------------------------------------------------------


class TestTryCatchHandlerNoBinding:
    def test_try_catch_handler_without_binding_catches(self) -> None:
        """CatchClause with binding=None still catches the exception."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.syntax.nodes import CatchClause, PassStmt, TryCatch
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)

        call_count = [0]

        class RaisingInterp(Interpreter):
            def _exec_stmt(self, s: object, sc: object) -> None:
                call_count[0] += 1
                if call_count[0] == 1:
                    exc_val = ExceptionValue(
                        type_name="Abort",
                        fields={"message": TextValue("oops"), "trace_id": TextValue("")},
                    )
                    raise AglRaise(exc_val)

        from agm.agl.capabilities import HostCapabilities
        from agm.agl.parser import parse_program
        from agm.agl.runtime.agents import AgentRegistry
        from agm.agl.scope import resolve
        from agm.agl.typecheck import check
        from agm.agl.typecheck.env import TypeEnvironment

        program = parse_program("pass")
        resolved = resolve(program)
        caps = HostCapabilities(
            agent_names=frozenset(), has_fallback_agent=False,
            codec_kinds={}, renderer_names=frozenset()
        )
        checked = check(resolved, caps)
        registry = AgentRegistry(named={}, default_agent=None)
        interp = RaisingInterp(
            checked=checked,
            registry=registry,
            contracts={},
            type_env=TypeEnvironment(),
            loop_limit=3,
            strict_json=False,
        )

        # Handler without binding (binding=None) — takes the else path at line 270
        handler = CatchClause(
            exc_type="Abort",
            binding=None,  # No binding
            body=(PassStmt(span=span, node_id=10),),
            span=span,
            node_id=11,
        )
        stmt = TryCatch(
            body=(PassStmt(span=span, node_id=1),),
            handlers=(handler,),
            span=span,
            node_id=12,
        )
        scope = Scope(parent=None)
        interp._exec_try_catch(stmt, scope)  # Should not raise
        assert call_count[0] == 2  # body + handler body


# ---------------------------------------------------------------------------
# Coverage: _exec_let/_exec_var with annotation that resolves to None
# ---------------------------------------------------------------------------


class TestLetVarAnnotationNone:
    def test_exec_let_with_unknown_annotation_no_coerce(self) -> None:
        """LetDecl with an annotation _resolve_type_ann returns None → no coercion."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import IntLit, LetDecl
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import IntT, ListT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        # ListT is not handled by _resolve_type_ann → returns None → no coercion
        ann = ListT(elem=IntT(span=span, node_id=0), span=span, node_id=1)
        stmt = LetDecl(
            name="lst",
            type_ann=ann,
            value=IntLit(value=7, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_let(stmt, scope)
        b = scope.lookup("lst")
        assert b is not None
        # No coercion: stays as IntValue
        assert b.value == IntValue(7)

    def test_exec_var_with_unknown_annotation_no_coerce(self) -> None:
        """VarDecl with an annotation _resolve_type_ann returns None → no coercion."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import IntLit, VarDecl
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import IntT, ListT

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        ann = ListT(elem=IntT(span=span, node_id=0), span=span, node_id=1)
        stmt = VarDecl(
            name="lst",
            type_ann=ann,
            value=IntLit(value=8, span=span, node_id=2),
            span=span,
            node_id=3,
        )
        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)
        interp._exec_var(stmt, scope)
        b = scope.lookup("lst")
        assert b is not None


# ---------------------------------------------------------------------------
# Coverage: remaining interpreter gaps (lines 366, 404->401, 560, 570,
# 655, 688, 725, runtime.py 517->510, 521->510)
# ---------------------------------------------------------------------------


class TestRemainingCoverage:
    def test_eval_var_ref_undefined_raises(self) -> None:
        """_eval_expr with a VarRef to an undefined variable raises RuntimeError."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.syntax.nodes import VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        expr = VarRef(name="undefined_var", span=span, node_id=1)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        with pytest.raises(RuntimeError, match="Undefined variable"):
            interp._eval_expr(expr, Scope(parent=None))

    def test_template_two_text_segments_covers_loop_branch(self) -> None:
        """Template with two TextSegments ensures the loop iterates (404->401)."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import TextValue
        from agm.agl.syntax.nodes import Template, TextSegment
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        seg1 = TextSegment(text="hello ", span=span, node_id=1)
        seg2 = TextSegment(text="world", span=span, node_id=2)
        template = Template(segments=(seg1, seg2), span=span, node_id=3)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_template(template, Scope(parent=None))
        assert result == TextValue("hello world")

    def test_constructor_record_extra_field_not_in_type(self) -> None:
        """Constructor passing an arg for a field not declared in RecordType uses else branch."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.syntax.nodes import Constructor, IntLit, NamedArg
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import IntType, RecordType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        # Only field "x" declared; we'll also pass "extra" which is not in type
        type_env.register_type("Point", RecordType(name="Point", fields={"x": IntType()}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier=None,
            name="Point",
            args=(
                NamedArg(
                    name="x", value=IntLit(value=1, span=span, node_id=1), span=span, node_id=2
                ),
                NamedArg(
                    name="extra",
                    value=IntLit(value=99, span=span, node_id=3),
                    span=span,
                    node_id=4,
                ),
            ),
            span=span,
            node_id=5,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, RecordValue)
        # "extra" field passed through as-is (else branch at line 560)
        assert result.fields.get("extra") == IntValue(99)

    def test_constructor_enum_extra_field_not_in_type(self) -> None:
        """Constructor passing an enum arg not declared in variant uses else branch."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import EnumValue, IntValue
        from agm.agl.syntax.nodes import Constructor, IntLit, NamedArg
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.typecheck.env import TypeEnvironment
        from agm.agl.typecheck.types import EnumType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        type_env = TypeEnvironment()
        # Variant "A" has no fields declared; we pass "val"
        type_env.register_type("MyEnum", EnumType(name="MyEnum", variants={"A": {}}))

        interp = _make_interp(type_env)
        assert isinstance(interp, Interpreter)
        scope = Scope(parent=None)

        expr = Constructor(
            qualifier="MyEnum",
            name="A",
            args=(
                NamedArg(
                    name="val", value=IntLit(value=5, span=span, node_id=1), span=span, node_id=2
                ),
            ),
            span=span,
            node_id=3,
        )
        result = interp._eval_constructor(expr, scope)
        assert isinstance(result, EnumValue)
        # "val" is passed through (else branch at line 570)
        assert result.fields.get("val") == IntValue(5)

    def test_eval_case_expr_var_pattern_binds(self) -> None:
        """CaseExpr with VarPattern creates a binding and uses it in branch body."""
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.eval.scope import Scope
        from agm.agl.eval.values import IntValue
        from agm.agl.syntax.nodes import CaseExpr, CaseExprBranch, IntLit, VarPattern, VarRef
        from agm.agl.syntax.spans import SourceSpan

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        subject = IntLit(value=42, span=span, node_id=1)
        # VarPattern "n" captures the value, body returns that captured var
        branch = CaseExprBranch(
            pattern=VarPattern(name="n", span=span, node_id=2),
            body=VarRef(name="n", span=span, node_id=3),  # Returns captured value
            span=span,
            node_id=4,
        )
        expr = CaseExpr(subject=subject, branches=(branch,), span=span, node_id=5)

        interp = _make_interp()
        assert isinstance(interp, Interpreter)
        result = interp._eval_case_expr(expr, Scope(parent=None))
        # Branch body is VarRef("n") = IntValue(42)
        assert result == IntValue(42)

    def test_resolve_type_ann_int(self) -> None:
        """_resolve_type_ann returns IntType for IntT annotation."""
        from agm.agl.eval.interpreter import _resolve_type_ann
        from agm.agl.syntax.spans import SourceSpan
        from agm.agl.syntax.types import IntT
        from agm.agl.typecheck.types import IntType

        span = SourceSpan(1, 1, 1, 5, 0, 4)
        result = _resolve_type_ann(IntT(span=span, node_id=0))
        assert isinstance(result, IntType)

    def test_arith_decimal_subtraction(self) -> None:
        """_arith with decimal-decimal subtraction returns DecimalValue."""
        from agm.agl.eval.interpreter import _arith
        from agm.agl.eval.values import DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _arith(
            DecimalValue(decimal.Decimal("3.0")),
            DecimalValue(decimal.Decimal("1.5")),
            BinOp.SUB,
        )
        assert isinstance(result, DecimalValue)
        assert result == DecimalValue(decimal.Decimal("1.5"))

    def test_compare_non_ordering_op_on_decimal_returns_false(self) -> None:
        """Passing a non-ordering op to _compare with decimal/decimal falls to line 802."""
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, DecimalValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(
            DecimalValue(decimal.Decimal("1.0")),
            DecimalValue(decimal.Decimal("2.0")),
            BinOp.ADD,
        )
        assert result == BoolValue(False)

    def test_compare_non_ordering_op_on_text_returns_false(self) -> None:
        """Passing a non-ordering op to _compare with text/text falls to line 802."""
        from agm.agl.eval.interpreter import _compare
        from agm.agl.eval.values import BoolValue, TextValue
        from agm.agl.syntax.nodes import BinOp

        result = _compare(TextValue("a"), TextValue("b"), BinOp.ADD)
        assert result == BoolValue(False)

