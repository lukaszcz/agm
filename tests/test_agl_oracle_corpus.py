"""Full-corpus legacy-vs-IR differential oracle for the M7 runtime switch."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import pytest

from tests.agl.oracle.harness import _normalize_value
from tests.test_agl_e2e import (
    AGL_DIR,
    PROGRAMS_DIR,
    ScriptedAgent,
    _agent_from_spec,
    _load_json,
    _run_program,
)


def _scenarios() -> list[Any]:
    params: list[Any] = []
    for program in sorted(PROGRAMS_DIR.rglob("*.agl")):
        sidecar = program.with_name(f"{program.stem}.scenarios.json")
        rel = program.relative_to(PROGRAMS_DIR).with_suffix("")
        for scenario in _load_json(sidecar)["scenarios"]:
            params.append(pytest.param(program, scenario, id=f"{rel}::{scenario['name']}"))
    return params


def _legacy_agents(scenario: dict[str, Any]) -> dict[str, ScriptedAgent]:
    return {
        name: _agent_from_spec(name, spec)
        for name, spec in scenario.get("agents", {}).items()
    }


def _legacy_runtime(scenario: dict[str, Any], agents: dict[str, ScriptedAgent]) -> Any:
    from agm.agl import WorkflowRuntime

    cfg = scenario.get("runtime", {})
    kwargs: dict[str, Any] = {}
    if "default_loop_limit" in cfg:
        kwargs["default_loop_limit"] = cfg["default_loop_limit"]
    if "default_strict_json" in cfg:
        kwargs["default_strict_json"] = cfg["default_strict_json"]
    if "ask" in agents:
        kwargs["default_agent"] = agents["ask"]
    runtime = WorkflowRuntime(**kwargs)
    for name, agent in agents.items():
        if name != "ask":
            runtime.register_agent(name, agent)
    return runtime


def _contracts(checked_modules: list[Any], codecs: dict[str, Any]) -> dict[int, Any]:
    from agm.agl.runtime.contract import materialize_contract

    result: dict[int, Any] = {}
    for checked in checked_modules:
        for node_id, spec in checked.contract_specs.items():
            result[node_id] = materialize_contract(spec, codecs)
    return result


def _params(checked: Any, raw_params: dict[str, object]) -> dict[str, Any]:
    from agm.agl.runtime.runtime import convert_param_value
    from agm.agl.syntax.nodes import ParamDecl

    result: dict[str, Any] = {}
    for item in checked.resolved.program.body.items:
        if isinstance(item, ParamDecl) and item.name in raw_params:
            typ = checked.type_env.get_binding_type(item.node_id)
            assert typ is not None
            result[item.name] = convert_param_value(item.name, raw_params[item.name], typ)
    return result


def _run_legacy(
    source: str, scenario: dict[str, Any]
) -> tuple[dict[str, Any] | None, Any | None, str, dict[str, ScriptedAgent]]:
    from agm.agl.eval.exceptions import AglRaise
    from agm.agl.eval.interpreter import Interpreter, execute_graph
    from agm.agl.eval.scope import Scope
    from agm.agl.modules.roots import RootSet
    from agm.agl.parser import parse_program
    from agm.agl.runtime.trace import noop_trace
    from agm.agl.scope import resolve
    from agm.agl.typecheck import check
    from agm.agl.typecheck.graph import check_graph

    agents = _legacy_agents(scenario)
    runtime = _legacy_runtime(scenario, agents)
    host = runtime.host_environment()
    raw_params = scenario.get("params", {})
    cfg = scenario.get("runtime", {})
    loop_limit = int(cfg.get("default_loop_limit", runtime.default_loop_limit))
    strict_json = bool(cfg.get("default_strict_json", runtime.default_strict_json))
    module_roots = scenario.get("module_roots", [])
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            if module_roots:
                roots = RootSet(
                    roots=frozenset(
                        (AGL_DIR / str(root)).resolve() for root in module_roots
                    )
                )
                prepared = runtime.prepare_program(source, entry_path=None, roots=roots)
                assert prepared.resolved_graph is not None
                checked_graph = check_graph(prepared.resolved_graph, host.capabilities)
                checked_entry = checked_graph.modules[checked_graph.entry_id]
                bindings = execute_graph(
                    checked_graph,
                    host.registry,
                    _contracts(list(checked_graph.modules.values()), host.codecs),
                    loop_limit=loop_limit,
                    strict_json=strict_json,
                    param_values=_params(checked_entry, raw_params),
                )
            else:
                checked = check(resolve(parse_program(source)), host.capabilities)
                root = Scope(parent=None)
                interpreter = Interpreter(
                    checked=checked,
                    registry=host.registry,
                    contracts=_contracts([checked], host.codecs),
                    type_env=checked.type_env,
                    loop_limit=loop_limit,
                    strict_json=strict_json,
                    source=source,
                    trace=noop_trace(),
                    param_values=_params(checked, raw_params),
                )
                interpreter.execute(root)
                bindings = root.snapshot()
        return bindings, None, output.getvalue(), agents
    except AglRaise as exc:
        return None, _normalize_value(exc.exc), output.getvalue(), agents


def _normalized_bindings(bindings: dict[str, Any]) -> dict[str, Any]:
    return {name: _normalize_value(value) for name, value in bindings.items()}


def _normalize_trace_text(value: str) -> str:
    return re.sub(r"\b[0-9a-f]{32}\b", "<trace-id>", value)


def _recording_process(commands: list[tuple[str, ...]]) -> Any:
    from agm.core.process import run_capture_result

    def recording(cmd: list[str], **kwargs: Any) -> Any:
        commands.append(tuple(cmd))
        return run_capture_result(cmd, **kwargs)

    return recording


@pytest.mark.oracle
@pytest.mark.parametrize(("program", "scenario"), _scenarios())
def test_full_corpus_oracle(program: Path, scenario: dict[str, Any]) -> None:
    from unittest.mock import patch

    source = program.read_text(encoding="utf-8")
    ir_output = io.StringIO()
    ir_exec_commands: list[tuple[str, ...]] = []
    with patch(
        "agm.core.process.run_capture_result",
        side_effect=_recording_process(ir_exec_commands),
    ):
        with contextlib.redirect_stdout(ir_output):
            ir_result, ir_agents = _run_program(source, scenario)

    if "host_error" in scenario["expect"]:
        assert not ir_result.ok
        assert all(not agent.prompts for agent in ir_agents.values())
        return

    legacy_exec_commands: list[tuple[str, ...]] = []
    with patch(
        "agm.core.process.run_capture_result",
        side_effect=_recording_process(legacy_exec_commands),
    ):
        legacy_bindings, legacy_error, legacy_stdout, legacy_agents = _run_legacy(
            source, scenario
        )
    assert legacy_exec_commands == ir_exec_commands
    assert _normalize_trace_text(legacy_stdout) == _normalize_trace_text(
        ir_output.getvalue()
    )
    assert {
        name: [_normalize_trace_text(prompt) for prompt in agent.prompts]
        for name, agent in legacy_agents.items()
    } == {
        name: [_normalize_trace_text(prompt) for prompt in agent.prompts]
        for name, agent in ir_agents.items()
    }

    if legacy_error is None:
        assert ir_result.error is None
        assert legacy_bindings is not None
        assert _normalized_bindings(legacy_bindings) == _normalized_bindings(
            ir_result.bindings
        )
    else:
        assert ir_result.error is not None
        assert legacy_error.display_name == ir_result.error.type_name
        legacy_message = legacy_error.fields["message"].value
        assert legacy_message == ir_result.error.fields["message"]
