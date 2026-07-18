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
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody
from agm.agl.lower import lower_module
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import (
    MatchCompiledModule,
    MatchCompiledProgram,
    compile_module_matches,
    compile_program_matches,
)
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import ModuleGraph, load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program
from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import ExceptionValue, Value
from agm.agl.typecheck import check_module
from agm.agl.typecheck.program import CheckedProgram, check_program
from agm.core.process import ProcessCaptureResult

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[2] / "stdlib"


def _compiled_program(source: str, *, caps: HostCapabilities | None = None) -> MatchCompiledModule:
    checked = check_module(resolve_module(parse_program(source)), caps or base_caps())
    result = compile_module_matches(checked)
    assert isinstance(result.compiled, MatchCompiledModule)
    return result.compiled


def _compiled_checked(checked: object) -> MatchCompiledModule | MatchCompiledProgram:
    from agm.agl.typecheck import CheckedModule

    if isinstance(checked, CheckedModule):
        result = compile_module_matches(checked)
        assert isinstance(result.compiled, MatchCompiledModule)
        return result.compiled
    assert isinstance(checked, CheckedProgram)
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    return result.compiled


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset((*paths, _REPO_STDLIB_ROOT)))


def base_caps() -> HostCapabilities:
    return HostCapabilities(
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
        }
    )


def extern_caps() -> HostCapabilities:
    base = base_caps()
    return HostCapabilities(supports_extern=True, codec_kinds=base.codec_kinds)


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
    compiled = _compiled_program(source, caps=caps)
    executable = lower_module(compiled, source_text=source, source_label="<ir-test>")
    params = _build_ir_param_values(executable, param_values) if param_values else None
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = IrInterpreter(executable, registry=registry, param_values=params).run()
    return result, output.getvalue()


def evaluate_ir(source: str, param_values: dict[str, Value] | None = None) -> dict[str, Value]:
    result, _ = _run_ir(source, param_values)
    return result


def evaluate_ir_output(source: str, param_values: dict[str, Value] | None = None) -> str:
    """Run the program through the IR pipeline and return its captured stdout."""
    _, output = _run_ir(source, param_values)
    return output


def evaluate_ir_raises(source: str, param_values: dict[str, Value] | None = None) -> ExceptionValue:
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


def write_companion_file(root: Path, dotted: str, source: str = "") -> Path:
    """Write *dotted*'s Python companion file (the ``.agl`` sibling's ``.py`` twin).

    Used by extern-def fixtures: any module declaring an extern needs a real
    companion file on disk before it can be loaded.
    """
    agl_path = root / ModuleId.from_dotted(dotted).relpath().replace("/", os.sep)
    py_path = agl_path.with_suffix(".py")
    py_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.write_text(source)
    return py_path


def _prepare_extern_program(
    source: str,
    companion_source: str,
    tmp_path: Path,
    *,
    caps: HostCapabilities | None = None,
) -> tuple[ExecutableProgram, ExternRegistry]:
    """Resolve + check + lower a single-module extern-declaring *source*.

    Writes *source* and *companion_source* as real sibling files on disk (an
    extern def needs a resolvable origin path, and the registry needs a real
    file to import), then builds an ``ExternRegistry`` populated the same way
    the pipeline wires one before evaluation — one ``load_companion`` per
    declaring module, mirroring ``pipeline._wire_extern_registry``.
    """
    entry_path = tmp_path / "entry.agl"
    entry_path.write_text(source)
    companion_path = tmp_path / "entry.py"
    companion_path.write_text(companion_source)

    resolved = resolve_module(parse_program(source), origin_path=entry_path)
    checked = check_module(resolved, caps or extern_caps())
    executable = lower_module(
        _compiled_checked(checked),
        source_text=source,
        source_label="<extern-ir-test>",
    )
    registry = ExternRegistry()
    loaded: set[ModuleId] = set()
    for desc in executable.functions.values():
        if not isinstance(desc.impl, ExternFunctionBody) or desc.module_id in loaded:
            continue
        registry.load_companion(desc.module_id, companion_path)
        loaded.add(desc.module_id)
    return executable, registry


def evaluate_ir_with_externs(
    source: str,
    companion_source: str,
    tmp_path: Path,
    *,
    param_values: dict[str, Value] | None = None,
    caps: HostCapabilities | None = None,
) -> tuple[dict[str, Value], str]:
    """Run a single-module program declaring ``extern def`` end to end.

    Returns ``(bindings, captured_stdout)``, mirroring ``_run_ir``.
    """
    executable, registry = _prepare_extern_program(source, companion_source, tmp_path, caps=caps)
    params = _build_ir_param_values(executable, param_values) if param_values else None
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = IrInterpreter(executable, param_values=params, extern_registry=registry).run()
    return result, output.getvalue()


def evaluate_ir_raises_with_externs(
    source: str,
    companion_source: str,
    tmp_path: Path,
    *,
    caps: HostCapabilities | None = None,
) -> ExceptionValue:
    executable, registry = _prepare_extern_program(source, companion_source, tmp_path, caps=caps)
    try:
        IrInterpreter(executable, extern_registry=registry).run()
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR extern program did not raise AglRaise")


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


def _checked(entry_source: str, modules: dict[str, str], tmp_path: Path) -> CheckedProgram:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for dotted, source in modules.items():
        write_module_file(root, dotted, source)
    graph = load_graph(entry_source, entry_path=None, roots=_roots(root))
    return check_program(resolve_program(graph), base_caps())


def evaluate_ir_graph(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> dict[str, Value]:
    checked = _checked(entry_source, modules, tmp_path)
    executable = lower_program(_compiled_checked(checked))
    result = IrInterpreter(executable).run()
    return result


def evaluate_ir_graph_raises(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> ExceptionValue:
    checked = _checked(entry_source, modules, tmp_path)
    executable = lower_program(_compiled_checked(checked))
    try:
        IrInterpreter(executable).run()
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR graph did not raise AglRaise")


def agent_caps(agent_names: frozenset[str], *, has_default: bool = False) -> HostCapabilities:
    base = base_caps()
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
        make_agent("__default__", default_responses) if default_responses is not None else None
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
    caps = agent_caps(agent_names or frozenset(scripts), has_default=has_default)
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
    caps = agent_caps(agent_names or frozenset(scripts), has_default=has_default)
    registry = _make_scripted_registry(scripts, default_responses=default_responses)
    try:
        _run_ir(source, caps=caps, registry=registry)
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR agent program did not raise AglRaise")


def shell_caps(
    *, agent_names: frozenset[str] = frozenset(), has_default: bool = False
) -> HostCapabilities:
    base = base_caps()
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
    result, _ = _run_ir_exec(source, shell, caps or shell_caps())
    return result


def evaluate_ir_raises_with_shell(
    source: str,
    commands: dict[str, ProcessCaptureResult],
    caps: HostCapabilities | None = None,
) -> ExceptionValue:
    try:
        _run_ir_exec(source, _scripted_shell(commands), caps or shell_caps())
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR shell program did not raise AglRaise")
