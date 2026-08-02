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
from agm.agl.ir.ids import AgentId, SourceId, SymbolId
from agm.agl.ir.program import ExecutableProgram, ExternFunctionBody, SourceFile
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.lowerer import _LinkState, _Lowerer, builtin_nominals_from_declarations
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import MatchCompiledModule, MatchCompiledProgram, compile_program_matches
from agm.agl.matchcompile.stage import _compile_owner_sites
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import ModuleGraph, load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.scope.program import resolve_program
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.values import ExceptionValue, Value
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram, check_program
from agm.core.process import ProcessCaptureResult
from agm.util.text import normalize_newlines
from tests.agl.module_graph import build_module_graph as _build_module_graph

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[2] / "stdlib"


def _checked_program(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> CheckedProgram:
    """Resolve and check *source* through a real module graph.

    Reuses :func:`tests.agl.module_graph._build_graph` (entry plus a
    process-cached ``std/core`` unless *default_stdlib* is ``False``) so this
    pays the same one-parse-per-process cost as the scope/typecheck unit
    helpers, then runs the real whole-program passes -- the configuration
    production always runs, and the only one under which the module
    loader's own checks (e.g. an extern's missing companion file) fire.
    """
    graph, _import_node_id = _build_module_graph(
        source, origin_path=origin_path, default_stdlib=default_stdlib
    )
    resolved_program = resolve_program(graph)
    return check_program(resolved_program, caps or base_caps())


def _compiled_program(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> MatchCompiledProgram:
    checked = _checked_program(
        source, caps=caps, default_stdlib=default_stdlib, origin_path=origin_path
    )
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    return result.compiled


def _compiled_checked(checked: CheckedProgram) -> MatchCompiledProgram:
    result = compile_program_matches(checked)
    assert isinstance(result.compiled, MatchCompiledProgram)
    return result.compiled


def compile_checked_module(checked: CheckedModule) -> MatchCompiledModule:
    """Reimplement the deleted ``compile_module_matches`` for tests that need a
    per-module compiled artifact from a hand-resolved or virtual-path
    ``CheckedModule`` with no real module graph behind it (see
    :func:`~tests.agl.module_graph.resolve_and_check_program_ast`).

    Production only ever compiles a whole program
    (:func:`~agm.agl.matchcompile.compile_program_matches`), so there is no
    surviving public per-module entry point; this reuses the same
    ``_compile_owner_sites`` building block that function calls once per
    module. Callers that need a rejected (non-exhaustive/refutable) result
    should call :func:`~agm.agl.matchcompile.stage._compile_owner_sites`
    themselves instead -- this helper asserts every site compiled cleanly.
    """
    sites, issues = _compile_owner_sites(checked)
    assert not issues
    return MatchCompiledModule(checked=checked, sites=sites)


def lower_compiled_module(
    compiled: MatchCompiledModule, *, source_text: str, source_label: str
) -> ExecutableProgram:
    """Reimplement the deleted ``lower_module`` for the same seam as
    :func:`compile_checked_module`: a per-module lowering, with no real
    module graph, for a test that needs to isolate the lowerer's own shape
    from ``lower_program``'s whole-program linking (or, forging a corrupt
    artifact, from ``lower_program``'s own module set).
    """
    checked = compiled.checked
    link = _LinkState(builtin_nominals=builtin_nominals_from_declarations({ENTRY_ID: checked}))
    source_id = SourceId(link.next_source)
    link.next_source += 1
    link.sources[source_id] = SourceFile(
        display_name=source_label, normalized_text=normalize_newlines(source_text)
    )
    lowerer = _Lowerer(checked, link, ENTRY_ID, source_id, source_text, compiled.sites)
    program = lowerer.lower()
    if self_validation_enabled():
        validate_ir(program)
    return program


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset((*paths, _REPO_STDLIB_ROOT)))


def base_caps() -> HostCapabilities:
    return HostCapabilities(
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}
            ),
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


def lower_ir(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> ExecutableProgram:
    """Resolve, check, compile, and lower *source* through the real
    program-level pipeline (entry plus ``std/core`` unless *default_stdlib*
    is ``False``), returning the linked ``ExecutableProgram`` without running
    it.

    Shared by every "golden lowering" test that inspects IR node shape
    directly rather than executed values -- ``lower_program`` (not the
    per-module ``lower_module``) is the pipeline production always runs, so
    this is what a source-level IR test should lower through too.
    """
    compiled = _compiled_program(
        source, caps=caps, default_stdlib=default_stdlib, origin_path=origin_path
    )
    return lower_program(compiled, _entry_source_text=source)


def _run_ir(
    source: str,
    param_values: dict[str, Value] | None = None,
    *,
    caps: HostCapabilities | None = None,
    registry: AgentRegistry | None = None,
    default_stdlib: bool = True,
) -> tuple[dict[str, Value], str]:
    executable = lower_ir(source, caps=caps, default_stdlib=default_stdlib)
    params = _build_ir_param_values(executable, param_values) if param_values else None
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = IrInterpreter(executable, registry=registry, param_values=params).run()
    return result, output.getvalue()


def evaluate_ir(
    source: str,
    param_values: dict[str, Value] | None = None,
    *,
    default_stdlib: bool = True,
) -> dict[str, Value]:
    result, _ = _run_ir(source, param_values, default_stdlib=default_stdlib)
    return result


def evaluate_ir_output(
    source: str,
    param_values: dict[str, Value] | None = None,
    *,
    default_stdlib: bool = True,
) -> str:
    """Run the program through the IR pipeline and return its captured stdout."""
    _, output = _run_ir(source, param_values, default_stdlib=default_stdlib)
    return output


def evaluate_ir_raises(
    source: str,
    param_values: dict[str, Value] | None = None,
    *,
    default_stdlib: bool = True,
) -> ExceptionValue:
    try:
        _run_ir(source, param_values, default_stdlib=default_stdlib)
    except AglRaise as exc:
        return exc.exc
    raise AssertionError("IR pipeline did not raise AglRaise")


def write_module_file(root: Path, module_path: str, source: str) -> Path:
    path = root / ModuleId.from_path(module_path).relpath().replace("/", os.sep)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def write_companion_file(root: Path, module_path: str, source: str = "") -> Path:
    """Write a module path's Python companion file (the ``.agl`` sibling's ``.py`` twin).

    Used by extern-def fixtures: any module declaring an extern needs a real
    companion file on disk before it can be loaded.
    """
    agl_path = root / ModuleId.from_path(module_path).relpath().replace("/", os.sep)
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
    """Resolve + check + lower an extern-declaring *source* through a real module graph.

    Writes *source* and *companion_source* as real sibling files on disk (an
    extern def needs a resolvable origin path, and the registry needs a real
    file to import) and resolves *source* at that real path -- the entry's
    real origin, so the module loader's own missing-companion check runs
    exactly as it does in production -- then builds an ``ExternRegistry``
    populated the same way the pipeline wires one before evaluation -- one
    ``load_companion`` per declaring module, mirroring
    ``pipeline._wire_extern_registry``.
    """
    entry_path = tmp_path / "entry.agl"
    entry_path.write_text(source)
    companion_path = tmp_path / "entry.py"
    companion_path.write_text(companion_source)

    executable = lower_ir(source, caps=caps or extern_caps(), origin_path=entry_path)
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


def make_graph_from_files(
    tmp_path: Path, modules: dict[str, str], *, default_stdlib: bool = True
) -> ModuleGraph:
    """Build a ModuleGraph via ``load_graph`` from a ``{name: source}`` dict.

    The key ``'entry'`` is used as the entry source; all other keys are written
    as ``.agl`` module files under a temp root. ``default_stdlib`` threads
    through to ``load_graph`` for a caller that needs a program without the
    shipped standard library.
    """
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    entry_source = modules.get("entry", "()")
    for module_path, source in modules.items():
        if module_path == "entry":
            continue
        write_module_file(root, module_path, source)
    return load_graph(
        entry_source, entry_path=None, roots=_roots(root), default_stdlib=default_stdlib
    )


def _checked(entry_source: str, modules: dict[str, str], tmp_path: Path) -> CheckedProgram:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for module_path, source in modules.items():
        write_module_file(root, module_path, source)
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

    named = {AgentId(name): make_agent(name, responses) for name, responses in scripts.items()}
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
