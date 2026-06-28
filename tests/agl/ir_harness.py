"""IR-only semantic test helpers for source, graph, agent, and shell programs."""

from __future__ import annotations

import contextlib
import io
import os
import unittest.mock
from collections.abc import Callable
from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.ir.ids import SymbolId
from agm.agl.ir.program import ExecutableProgram
from agm.agl.lower import lower_program
from agm.agl.lower.graph import lower_graph
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import ModuleGraph, load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import ExceptionValue, Value
from agm.agl.typecheck import check
from agm.agl.typecheck.graph import check_graph
from agm.core.process import ProcessCaptureResult

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[2] / "stdlib"


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset((*paths, _REPO_STDLIB_ROOT)))


def m2_caps() -> HostCapabilities:
    return HostCapabilities(
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        }
    )


def _build_ir_param_values(
    executable: ExecutableProgram, param_values: dict[str, Value]
) -> dict[SymbolId, Value]:
    by_name = {param.public_name: param.symbol for param in executable.params}
    return {by_name[name]: value for name, value in param_values.items()}


def _run_ir(
    source: str,
    param_values: dict[str, Value] | None = None,
    *,
    caps: HostCapabilities | None = None,
    registry: AgentRegistry | None = None,
) -> tuple[dict[str, Value], str]:
    checked = check(resolve(parse_program(source)), caps or m2_caps())
    executable = lower_program(
        checked, source_text=source, source_label="<ir-test>", validate=True
    )
    params = _build_ir_param_values(executable, param_values) if param_values else None
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = IrInterpreter(
            executable, registry=registry, param_values=params
        ).run()
    return result, output.getvalue()


def evaluate_ir(
    source: str, param_values: dict[str, Value] | None = None
) -> dict[str, Value]:
    result, _ = _run_ir(source, param_values)
    return result


def evaluate_ir_output(source: str, param_values: dict[str, Value] | None = None) -> str:
    """Run the program through the IR pipeline and return its captured stdout."""
    _, output = _run_ir(source, param_values)
    return output


def evaluate_ir_raises(
    source: str, param_values: dict[str, Value] | None = None
) -> ExceptionValue:
    try:
        _run_ir(source, param_values)
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR pipeline did not raise AglRaise")


def write_module_file(root: Path, dotted: str, source: str) -> Path:
    path = root / ModuleId.from_dotted(dotted).relpath().replace("/", os.sep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def make_graph_from_files(tmp_path: Path, modules: dict[str, str]) -> ModuleGraph:
    """Build a ModuleGraph via ``load_graph`` from a ``{name: source}`` dict.

    The key ``'entry'`` is used as the entry source; all other keys are written
    as ``.agl`` module files under a temp root.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    entry_source = modules.get("entry", "()")
    for dotted, source in modules.items():
        if dotted == "entry":
            continue
        write_module_file(root, dotted, source)
    return load_graph(entry_source, entry_path=None, roots=_roots(root))


def _checked_graph(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> object:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for dotted, source in modules.items():
        write_module_file(root, dotted, source)
    graph = load_graph(entry_source, entry_path=None, roots=_roots(root))
    return check_graph(resolve_graph(graph), m2_caps())


def evaluate_ir_graph(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> dict[str, Value]:
    executable = lower_graph(_checked_graph(entry_source, modules, tmp_path), validate=True)
    result = IrInterpreter(executable).run()
    return result


def evaluate_ir_graph_raises(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> ExceptionValue:
    executable = lower_graph(_checked_graph(entry_source, modules, tmp_path), validate=True)
    try:
        IrInterpreter(executable).run()
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR graph did not raise AglRaise")


def m6b_caps(agent_names: frozenset[str], *, has_default: bool = False) -> HostCapabilities:
    base = m2_caps()
    return HostCapabilities(
        agent_names=agent_names,
        has_default_agent=has_default,
        codec_kinds=base.codec_kinds,
    )


def _make_scripted_registry(
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
    call_log: list[tuple[str, str]] | None = None,
) -> AgentRegistry:
    def make_agent(name: str, responses: list[str]) -> AgentFn:
        remaining = iter(responses)

        def agent(request: AgentRequest) -> AgentResponse:
            if call_log is not None:
                call_log.append((name, request.prompt))
            return AgentResponse(content=next(remaining))

        return agent

    named = {name: make_agent(name, responses) for name, responses in scripts.items()}
    default = (
        make_agent("__default__", default_responses)
        if default_responses is not None
        else None
    )
    return AgentRegistry(named=named, default_agent=default)


def evaluate_ir_with_agents(
    source: str,
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
    agent_names: frozenset[str] | None = None,
    has_default: bool = False,
) -> dict[str, Value]:
    caps = m6b_caps(agent_names or frozenset(scripts), has_default=has_default)
    registry = _make_scripted_registry(scripts, default_responses=default_responses)
    result, _ = _run_ir(source, caps=caps, registry=registry)
    return result


def evaluate_ir_raises_with_agents(
    source: str,
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
    agent_names: frozenset[str] | None = None,
    has_default: bool = False,
) -> ExceptionValue:
    caps = m6b_caps(agent_names or frozenset(scripts), has_default=has_default)
    registry = _make_scripted_registry(scripts, default_responses=default_responses)
    try:
        _run_ir(source, caps=caps, registry=registry)
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR agent program did not raise AglRaise")


def m6c_caps(
    *, agent_names: frozenset[str] = frozenset(), has_default: bool = False
) -> HostCapabilities:
    base = m2_caps()
    return HostCapabilities(
        agent_names=agent_names,
        has_default_agent=has_default,
        supports_shell_exec=True,
        codec_kinds=base.codec_kinds,
    )


def _scripted_shell(
    commands: dict[str, ProcessCaptureResult], *, cmd_log: list[str] | None = None
) -> Callable[..., ProcessCaptureResult]:
    def run(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        del idle_timeout, isolate_process_group
        command = args[2]
        if cmd_log is not None:
            cmd_log.append(command)
        return commands[command]

    return run


def _run_ir_exec(
    source: str,
    shell_fake: Callable[..., ProcessCaptureResult],
    caps: HostCapabilities,
) -> tuple[dict[str, Value], str]:
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell_fake):
        return _run_ir(source, caps=caps)


def evaluate_ir_with_shell(
    source: str,
    commands: dict[str, ProcessCaptureResult],
    caps: HostCapabilities | None = None,
    *,
    cmd_log_ir: list[str] | None = None,
) -> dict[str, Value]:
    shell = _scripted_shell(commands, cmd_log=cmd_log_ir)
    result, _ = _run_ir_exec(source, shell, caps or m6c_caps())
    return result


def evaluate_ir_raises_with_shell(
    source: str,
    commands: dict[str, ProcessCaptureResult],
    caps: HostCapabilities | None = None,
) -> ExceptionValue:
    try:
        _run_ir_exec(source, _scripted_shell(commands), caps or m6c_caps())
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR shell program did not raise AglRaise")
