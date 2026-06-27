"""End-to-end behavior tests for AgL v2 (the `agm exec` workflow DSL).

This suite is the TDD specification for the AgL implementation. Every
tests/agl/programs/**/*.agl file is a complete AgL v2 program executed under
each scenario in its sidecar `<name>.scenarios.json`; every
tests/agl/rejections/**/*.agl file is an invalid program that the static
pipeline must reject before executing anything. The data format is documented
in tests/agl/README.md.

Public contract exercised here (notes/PLAN_DSL.md §9, notes/dsl_design.md §7.6):

    from agm.agl import PipelineDriver

    runtime = PipelineDriver(
        default_strict_json=False,  # lenient JSON recovery is the default (design §2.8)
        default_agent=fn,           # the built-in `ask` agent (a host callable;
                                    # `ask` cannot be registered by name)
    )
    runtime.register_agent(name, fn)   # fn(request) -> str; request.prompt is the
                                       # rendered user prompt (design §7.5)
    result = runtime.run(source, param_values={...})

RunResult surface asserted:

    result.ok           True iff static checks and param validation passed and
                        no uncaught AgL exception was raised
    result.diagnostics  pre-execution failures (static errors, param
                        validation), each with `.message: str` and
                        `.line: int` (1-based source line)
    result.error        the uncaught AgL exception or None, exposing
                        `.type_name: str` and `.fields` — a mapping of the
                        exception's declared fields (including "message") to
                        JSON-shaped Python values

`print` writes to the process stdout (captured with capsys).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

AGL_DIR = Path(__file__).parent / "agl"
PROGRAMS_DIR = AGL_DIR / "programs"
REJECTIONS_DIR = AGL_DIR / "rejections"
REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"


def _load_json(path: Path) -> Any:
    # parse_float=Decimal: AgL has no binary floats (design §5.1); scenario
    # params and expected exception fields must round-trip decimals exactly.
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


@dataclass
class ScriptedAgent:
    """Replays a scenario's scripted responses and records rendered prompts."""

    name: str
    responses: list[str]
    repeat_last: bool = False
    prompts: list[str] = field(default_factory=list)
    overflowed: bool = False

    def __call__(self, request: Any) -> str:
        self.prompts.append(request.prompt)
        index = len(self.prompts) - 1
        if index < len(self.responses):
            return self.responses[index]
        if self.repeat_last and self.responses:
            return self.responses[-1]
        self.overflowed = True
        return ""


def _agent_from_spec(name: str, spec: Any) -> ScriptedAgent:
    if isinstance(spec, list):
        return ScriptedAgent(name=name, responses=[str(r) for r in spec])
    return ScriptedAgent(
        name=name,
        responses=[str(r) for r in spec["responses"]],
        repeat_last=bool(spec.get("repeat_last", False)),
    )


def _run_program(source: str, scenario: dict[str, Any]) -> tuple[Any, dict[str, ScriptedAgent]]:
    from agm.agl import PipelineDriver

    agents = {
        name: _agent_from_spec(name, spec) for name, spec in scenario.get("agents", {}).items()
    }
    kwargs: dict[str, Any] = {}
    runtime_cfg = scenario.get("runtime", {})
    if "default_call_depth_limit" in runtime_cfg:
        kwargs["default_call_depth_limit"] = runtime_cfg["default_call_depth_limit"]
    if "default_strict_json" in runtime_cfg:
        kwargs["default_strict_json"] = runtime_cfg["default_strict_json"]
    if "ask" in agents:
        kwargs["default_agent"] = agents["ask"]
    runtime = PipelineDriver(**kwargs)
    for name, agent in agents.items():
        if name != "ask":
            runtime.register_agent(name, agent)
    module_roots = scenario.get("module_roots", [])
    if module_roots:
        from agm.agl.modules.roots import RootSet

        roots = RootSet(
            roots=frozenset(
                {
                    *((AGL_DIR / str(root)).resolve() for root in module_roots),
                    REPO_STDLIB_ROOT,
                }
            )
        )
        prepared = PipelineDriver.prepare_program(source, entry_path=None, roots=roots)
        result = runtime.run_prepared_graph(
            prepared, param_values=scenario.get("params", {})
        )
    else:
        result = runtime.run(source, param_values=scenario.get("params", {}))
    return result, agents


def _assert_host_error(
    result: Any, agents: dict[str, ScriptedAgent], spec: dict[str, Any]
) -> None:
    assert not result.ok, "expected the run to fail param validation"
    assert result.error is None, "param validation failure is not an AgL exception"
    messages = " | ".join(d.message for d in result.diagnostics)
    for needle in spec.get("message_contains", []):
        assert needle.lower() in messages.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {messages!r}"
        )
    for agent in agents.values():
        assert agent.prompts == [], (
            f"agent {agent.name!r} was called despite param validation failing"
        )


def _assert_outcome(result: Any, expect: dict[str, Any]) -> None:
    diags = " | ".join(d.message for d in result.diagnostics)
    assert list(result.diagnostics) == [], f"unexpected static diagnostics: {diags}"
    if "raises" in expect:
        spec = expect["raises"]
        assert result.error is not None, f"expected uncaught {spec['type']}, got none"
        assert result.error.type_name == spec["type"]
        for key, value in spec.get("fields", {}).items():
            actual = result.error.fields[key]
            assert actual == value, f"{spec['type']}.{key}: expected {value!r}, got {actual!r}"
        message = str(result.error.fields.get("message", ""))
        for needle in spec.get("message_contains", []):
            assert needle in message, f"{needle!r} not in message {message!r}"
        assert not result.ok
    else:
        if result.error is not None:
            raise AssertionError(
                f"unexpected uncaught {result.error.type_name}: {result.error.fields!r}"
            )
        assert result.ok


def _assert_output(out: str, expect: dict[str, Any]) -> None:
    if "stdout" in expect:
        assert out == expect["stdout"]
    for needle in expect.get("stdout_contains", []):
        assert needle in out, f"{needle!r} not in stdout {out!r}"
    for needle in expect.get("stdout_not_contains", []):
        assert needle not in out, f"{needle!r} unexpectedly in stdout {out!r}"


def _assert_calls(agents: dict[str, ScriptedAgent], expect: dict[str, Any]) -> None:
    for name, agent in agents.items():
        assert not agent.overflowed, f"agent {name!r} was called more times than scripted"
    expected_calls = expect.get("calls", {})
    assert set(expected_calls) == set(agents), (
        "every scripted agent must have an exact call-count assertion; "
        f"expected entries for {sorted(agents)}, got {sorted(expected_calls)}"
    )
    for name, agent in agents.items():
        count = expected_calls[name]
        actual = len(agent.prompts)
        assert actual == count, f"agent {name!r}: expected {count} calls, got {actual}"
    for spec in expect.get("prompts", []):
        prompts = agents[spec["agent"]].prompts
        call = spec["call"]
        assert call < len(prompts), (
            f"agent {spec['agent']!r} made only {len(prompts)} calls, no call {call}"
        )
        prompt = prompts[call]
        if "equals" in spec:
            assert prompt == spec["equals"]
        for needle in spec.get("contains", []):
            assert needle in prompt, f"{needle!r} not in prompt {prompt!r}"
        for needle in spec.get("not_contains", []):
            assert needle not in prompt, f"{needle!r} unexpectedly in prompt {prompt!r}"


def _scenario_params() -> list[Any]:
    params: list[Any] = []
    for program in sorted(PROGRAMS_DIR.rglob("*.agl")):
        sidecar = program.with_name(program.stem + ".scenarios.json")
        rel = program.relative_to(PROGRAMS_DIR).with_suffix("")
        for scenario in _load_json(sidecar)["scenarios"]:
            params.append(pytest.param(program, scenario, id=f"{rel}::{scenario['name']}"))
    return params


def _rejection_params() -> list[Any]:
    return [
        pytest.param(program, id=str(program.relative_to(REJECTIONS_DIR).with_suffix("")))
        for program in sorted(REJECTIONS_DIR.rglob("*.agl"))
    ]


@pytest.mark.parametrize(("program", "scenario"), _scenario_params())
def test_program_scenario(
    program: Path, scenario: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    result, agents = _run_program(program.read_text(encoding="utf-8"), scenario)
    out = capsys.readouterr().out
    expect = scenario["expect"]
    if "host_error" in expect:
        _assert_host_error(result, agents, expect["host_error"])
        return
    _assert_outcome(result, expect)
    _assert_output(out, expect)
    _assert_calls(agents, expect)


@pytest.mark.parametrize("program", _rejection_params())
def test_static_rejection(program: Path) -> None:
    from agm.agl import PipelineDriver

    expect = _load_json(program.with_name(program.stem + ".expect.json"))["diagnostic"]
    result = PipelineDriver().run(program.read_text(encoding="utf-8"), param_values={})
    assert not result.ok, "expected the program to be rejected statically"
    assert result.error is None, "static rejection must happen before execution"
    diagnostics = list(result.diagnostics)
    assert diagnostics, "expected at least one diagnostic"
    if "line" in expect:
        lines = [d.line for d in diagnostics]
        assert expect["line"] in lines, f"no diagnostic on line {expect['line']}; lines: {lines}"
    joined = " | ".join(d.message for d in diagnostics)
    for needle in expect.get("message_contains", []):
        assert needle.lower() in joined.lower(), (
            f"no diagnostic mentions {needle!r}; diagnostics: {joined!r}"
        )
