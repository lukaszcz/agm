"""IR-only semantic test helpers for source, graph, agent, and shell programs."""

from __future__ import annotations

import contextlib
import io
import os
import time
import unittest.mock
from collections.abc import Callable
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.ir.nodes import IrBlock, IrConstUnit, IrExpr
from agm.agl.ir.program import ExecutableProgram, IrFunctionBody, ValueDescriptors
from agm.agl.lexer import spaced_qualifier_collector
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import MatchCompiledModule, MatchCompiledProgram, compile_program_matches
from agm.agl.matchcompile.stage import _compile_owner_sites
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import ModuleGraph, build_repl_graph, parse_entry_module
from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program_seeded
from agm.agl.pipeline import PipelineDriver, RunError, RunResult
from agm.agl.runtime import externs
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.boundary import encode_boundary_value
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import ScopeNode
from agm.agl.semantics.values import ContractValue, Value
from agm.agl.typecheck.env import CheckedModule
from agm.agl.typecheck.program import CheckedProgram, check_program
from agm.core.process import ProcessCaptureResult
from tests._agl_helpers import agl_roots, run_inline_command
from tests.agl.module_graph import build_module_graph, build_module_graph_from_program, load_graph

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[2] / "packages" / "stdlib"


def _checked_program(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> CheckedProgram:
    """Resolve and check *source* through a real module graph.

    Reuses :func:`tests.agl.module_graph._build_graph` (entry plus a
    process-cached ``std/prelude`` unless *default_stdlib* is ``False``) so this
    pays the same one-parse-per-process cost as the scope/typecheck unit
    helpers, then runs the real whole-program passes -- the configuration
    production always runs, and the only one under which the module
    loader's own checks (e.g. an extern's missing companion file) fire.
    """
    graph, _import_node_id = build_module_graph(
        source, origin_path=origin_path, default_stdlib=default_stdlib
    )
    resolved_program = resolve_program(graph)
    return check_program(resolved_program, caps or base_caps())


def _checked_inline_program(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> CheckedProgram:
    """Compile test-only inline source as ``agm exec -c`` does.

    Raw lowering helpers intentionally do not call this: shape tests must
    supply static-root source explicitly.
    """
    parsed = parse_entry_module(source, entry_path=origin_path, inline_command=True)
    graph, _import_node_id = build_module_graph_from_program(
        parsed.program,
        next_node_id=parsed.next_id,
        origin_path=origin_path,
        default_stdlib=default_stdlib,
        spaced_qualifiers=parsed.spaced_qualifiers,
    )
    return check_program(resolve_program(graph), caps or base_caps())


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


def _compiled_inline_program(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> MatchCompiledProgram:
    """Compile a statement-oriented test workflow as ``agm exec -c`` does."""
    result = compile_program_matches(
        _checked_inline_program(
            source, caps=caps, default_stdlib=default_stdlib, origin_path=origin_path
        )
    )
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


def lower_compiled_module(compiled: MatchCompiledModule, *, source_text: str) -> ExecutableProgram:
    """Lower a hand-built single-module artifact through ``lower_program``.

    This test seam is for checked modules with no loadable source graph, such
    as deliberately corrupt artifacts used by IR validation tests. It wraps
    the module in the smallest whole-program artifact rather than retaining a
    second lowering entry point.
    """
    checked = compiled.checked
    checked_program = CheckedProgram(
        modules={ENTRY_ID: checked},
        entry_id=ENTRY_ID,
        program_type_table={},
        warnings=(),
    )
    program = MatchCompiledProgram(
        checked=checked_program,
        sites_by_module={ENTRY_ID: compiled.sites},
    )
    return lower_program(program, _entry_source_text=source_text)


def inline_main_items(program: ExecutableProgram) -> tuple[IrExpr, ...]:
    """Return source items lowered into an inline command's synthetic ``main``.

    The synthetic program function has one compiler-added final unit result;
    this view removes that implementation detail while retaining an explicit
    trailing ``()`` written by the test source.
    """
    assert program.synthetic_main_symbol is not None
    function_id = program.program_functions[program.synthetic_main_symbol]
    impl = program.functions[function_id].impl
    assert isinstance(impl, IrFunctionBody)
    assert isinstance(impl.body, IrBlock)
    items = impl.body.items
    if items and isinstance(items[-1], IrConstUnit):
        return items[:-1]
    return items


def nominal_id_for(program: ExecutableProgram, display_name: str) -> NominalId:
    """Return the ``NominalId`` a lowered *program* uses for *display_name*.

    Nominal identity is an opaque per-declaration handle (see
    ``agm.agl.ir.ids.NominalId``), so a test asserting against lowered IR
    output cannot construct the expected id from a name and scope path: it
    looks the identity up from the program's own ``nominals`` table by the
    declaration's scoped source spelling (``NominalDescriptor.display_name``).
    Raises ``AssertionError`` if no descriptor carries that spelling, or if
    more than one does (ambiguous).
    """
    matches = [
        nominal for nominal, desc in program.nominals.items() if desc.display_name == display_name
    ]
    assert matches, f"no nominal with display_name {display_name!r} in program.nominals"
    assert len(matches) == 1, (
        f"ambiguous display_name {display_name!r}: {len(matches)} nominals match"
    )
    return matches[0]


def _roots(*paths: Path, include_stdlib: bool = True) -> RootSet:
    return agl_roots(*paths, include_stdlib=include_stdlib)


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


def lower_ir(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> ExecutableProgram:
    """Resolve, check, compile, and lower *source* through the real
    program-level pipeline (entry plus ``std/prelude`` unless *default_stdlib*
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


def lower_inline_ir(
    source: str,
    *,
    caps: HostCapabilities | None = None,
    default_stdlib: bool = True,
    origin_path: Path | None = None,
) -> ExecutableProgram:
    return lower_program(
        _compiled_inline_program(
            source, caps=caps, default_stdlib=default_stdlib, origin_path=origin_path
        ),
        _entry_source_text=source,
    )


def run_inline_ir(
    source: str,
    *,
    agent_dispatcher: AgentFn | None = None,
    default_stdlib: bool = True,
    entry_path: Path | None = None,
    roots: RootSet | None = None,
    process_environment: dict[str, str] | None = None,
    shell_exec_timeout: float | None = None,
) -> tuple[RunResult, str]:
    """Run *source* as ``agm exec -c`` through :class:`PipelineDriver`, capturing stdout.

    The driver receives only the test's own agent dispatcher; without one, an
    agent call fails instead of reaching a real agent.
    """
    runtime = PipelineDriver(
        agent_dispatcher=agent_dispatcher, shell_exec_timeout=shell_exec_timeout
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = run_inline_command(
            runtime,
            source,
            roots=_roots() if roots is None else roots,
            default_stdlib=default_stdlib,
            entry_path=entry_path,
            process_environment=process_environment,
        )
    return result, output.getvalue()


def completed_bindings(run: tuple[RunResult, str]) -> dict[str, Value]:
    """Return the bindings of a run that must have succeeded."""
    result, _ = run
    assert result.ok, (result.diagnostics, result.error)
    return result.bindings


def uncaught_error(run: tuple[RunResult, str]) -> RunError:
    """Return the uncaught AgL exception a run must have ended with."""
    result, _ = run
    assert result.error is not None, (result.diagnostics, result.bindings)
    return result.error


def evaluate_ir(source: str, *, default_stdlib: bool = True) -> dict[str, Value]:
    """Run a statement-oriented inline command and return module bindings."""
    return completed_bindings(run_inline_ir(source, default_stdlib=default_stdlib))


def evaluate_ir_output(source: str, *, default_stdlib: bool = True) -> str:
    """Run the program through the IR pipeline and return its captured stdout."""
    run = run_inline_ir(source, default_stdlib=default_stdlib)
    completed_bindings(run)
    return run[1]


def evaluate_ir_raises(source: str, *, default_stdlib: bool = True) -> RunError:
    """Run a program that must end with an uncaught AgL exception."""
    return uncaught_error(run_inline_ir(source, default_stdlib=default_stdlib))


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


def label_crossing_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encode each ``ContractValue`` crossing into an extern as ``contract-<id>`` text."""

    def encode(value: Value, descriptors: ValueDescriptors) -> object:
        if isinstance(value, ContractValue):
            return f"contract-{value.contract_id.value}"
        return encode_boundary_value(value, descriptors)

    monkeypatch.setattr(externs, "encode_boundary_value", encode)


def age_file(path: Path, *, seconds: int = 3600) -> None:
    """Back-date *path*'s mtime *seconds* before now, well past the write-settledness window."""
    stamp = time.time_ns() - seconds * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


def _write_extern_entry(source: str, companion_source: str, tmp_path: Path) -> Path:
    """Write an extern-declaring entry and its Python companion as real sibling files.

    An extern needs a resolvable origin path, so the loader's missing-companion
    check and the driver's companion import both run as in production.
    """
    entry_path = tmp_path / "entry.agl"
    entry_path.write_text(source)
    (tmp_path / "entry.py").write_text(companion_source)
    return entry_path


def lower_extern_program(source: str, companion_source: str, tmp_path: Path) -> ExecutableProgram:
    """Lower an extern-declaring *source* at its real entry path without running it."""
    entry_path = _write_extern_entry(source, companion_source, tmp_path)
    return lower_inline_ir(source, caps=extern_caps(), origin_path=entry_path)


def evaluate_ir_with_externs(
    source: str, companion_source: str, tmp_path: Path
) -> tuple[dict[str, Value], str]:
    """Run a single-module program declaring ``extern def``; return bindings and stdout."""
    run = run_inline_ir(source, entry_path=_write_extern_entry(source, companion_source, tmp_path))
    return completed_bindings(run), run[1]


def evaluate_ir_raises_with_externs(source: str, companion_source: str, tmp_path: Path) -> RunError:
    """Run an extern-declaring program that must end with an uncaught AgL exception."""
    return uncaught_error(
        run_inline_ir(source, entry_path=_write_extern_entry(source, companion_source, tmp_path))
    )


def _write_module_root(tmp_path: Path, modules: dict[str, str]) -> Path:
    """Write every module except ``entry`` under ``tmp_path/root`` and return that root."""
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    for module_path, source in modules.items():
        if module_path != "entry":
            write_module_file(root, module_path, source)
    return root


def make_repl_graph_from_files(
    tmp_path: Path, modules: dict[str, str], *, default_stdlib: bool = True
) -> ModuleGraph:
    """Build a raw incremental-REPL graph plus file-backed imports.

    Unlike :func:`make_inline_graph_from_files`, the entry stays at the REPL
    root. This is the appropriate seam for resolver tests that inspect or
    reject root expressions rather than model an ``agm exec -c`` invocation.
    """
    root = _write_module_root(tmp_path, modules)
    entry_source = modules.get("entry", "()")
    with spaced_qualifier_collector() as spaced_qualifiers:
        program, next_node_id = parse_program_seeded(entry_source, start_id=0, resolve_infix=False)
    graph, _next_id, _new_modules = build_repl_graph(
        program,
        next_node_id,
        path=None,
        cached={},
        roots=_roots(root, include_stdlib=default_stdlib),
        default_stdlib=default_stdlib,
        spaced_qualifiers=tuple(spaced_qualifiers),
        source_text=entry_source,
    )
    # ``resolve_program`` treats a supplied parent scope as the real REPL
    # entry marker, enabling root statements without affecting imported files.
    return graph


def resolve_repl_graph(graph: ModuleGraph):
    """Resolve a graph whose entry is an incremental REPL snippet."""
    return resolve_program(
        graph, entry_parent_scope=ScopeNode(node_id=-1, parent=None, scope_path=())
    )


def make_inline_graph_from_files(
    tmp_path: Path, modules: dict[str, str], *, default_stdlib: bool = True
) -> ModuleGraph:
    """Build a test-only ``agm exec -c`` graph plus file-backed imports.

    The ``entry`` value follows inline-command semantics; every other value
    is a static library file. Use :func:`make_file_graph_from_files` when the
    entry itself is a file-style program with an explicit entry declaration.
    """
    root = _write_module_root(tmp_path, modules)
    entry_source = modules.get("entry", "()")
    parsed = parse_entry_module(entry_source, entry_path=None, inline_command=True)
    graph, _next_id, _new_modules = build_repl_graph(
        parsed.program,
        parsed.next_id,
        path=None,
        cached={},
        roots=_roots(root, include_stdlib=default_stdlib),
        default_stdlib=default_stdlib,
        spaced_qualifiers=parsed.spaced_qualifiers,
        source_text=entry_source,
    )
    return graph


def make_file_graph_from_files(
    tmp_path: Path, modules: dict[str, str], *, default_stdlib: bool = True
) -> ModuleGraph:
    """Build a file-style graph from explicit entry-program source and imports."""
    root = _write_module_root(tmp_path, modules)
    entry_source = modules.get("entry", "()")
    return load_graph(
        entry_source,
        entry_path=None,
        roots=_roots(root, include_stdlib=default_stdlib),
        default_stdlib=default_stdlib,
    )


# Existing compiler fixtures use inline snippets. New call sites must select
# the explicitly named inline, REPL, or file helper above.
make_graph_from_files = make_inline_graph_from_files


def _checked(entry_source: str, modules: dict[str, str], tmp_path: Path) -> CheckedProgram:
    graph = make_inline_graph_from_files(tmp_path, {"entry": entry_source, **modules})
    return check_program(resolve_program(graph), base_caps())


def evaluate_ir_graph(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> dict[str, Value]:
    """Run an inline entry that imports file-backed *modules*; return its bindings."""
    roots = _roots(_write_module_root(tmp_path, modules))
    return completed_bindings(run_inline_ir(entry_source, roots=roots))


def evaluate_ir_graph_raises(
    entry_source: str, modules: dict[str, str], tmp_path: Path
) -> RunError:
    """Run an inline entry importing *modules* that must end with an uncaught AgL exception."""
    roots = _roots(_write_module_root(tmp_path, modules))
    return uncaught_error(run_inline_ir(entry_source, roots=roots))


def agent_caps() -> HostCapabilities:
    base = base_caps()
    return HostCapabilities(codec_kinds=base.codec_kinds)


def _make_scripted_registry(
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
    call_log: list[tuple[str, str]] | None = None,
) -> AgentFn:
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

    def dispatch(request: AgentRequest) -> AgentResponse:
        from agm.agent.spec import AgentCommand

        if not isinstance(request.agent, AgentCommand):
            assert default is not None
            return default(request)
        return named[request.agent.command](request)

    return dispatch


def evaluate_ir_with_agents(
    source: str,
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
) -> dict[str, Value]:
    agent_dispatcher = _make_scripted_registry(scripts, default_responses=default_responses)
    return completed_bindings(run_inline_ir(source, agent_dispatcher=agent_dispatcher))


def evaluate_ir_raises_with_agents(
    source: str,
    scripts: dict[str, list[str]],
    *,
    default_responses: list[str] | None = None,
) -> RunError:
    agent_dispatcher = _make_scripted_registry(scripts, default_responses=default_responses)
    return uncaught_error(run_inline_ir(source, agent_dispatcher=agent_dispatcher))


def shell_caps() -> HostCapabilities:
    base = base_caps()
    return HostCapabilities(supports_shell_exec=True, codec_kinds=base.codec_kinds)


def _scripted_shell(
    commands: dict[str, ProcessCaptureResult], *, cmd_log: list[str] | None = None
) -> Callable[..., ProcessCaptureResult]:
    def run(
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        del idle_timeout, cwd, env, isolate_process_group
        command = args[2]
        if cmd_log is not None:
            cmd_log.append(command)
        return commands[command]

    return run


def run_inline_ir_with_shell(
    source: str,
    shell_fake: Callable[..., ProcessCaptureResult],
    *,
    process_environment: dict[str, str] | None = None,
    shell_exec_timeout: float | None = None,
) -> tuple[RunResult, str]:
    """Run *source* like :func:`run_inline_ir` with every shell process faked by *shell_fake*."""
    with unittest.mock.patch("agm.core.process.run_capture_result", side_effect=shell_fake):
        return run_inline_ir(
            source,
            process_environment=process_environment,
            shell_exec_timeout=shell_exec_timeout,
        )


def evaluate_ir_with_shell(
    source: str,
    commands: dict[str, ProcessCaptureResult],
    *,
    cmd_log_ir: list[str] | None = None,
) -> dict[str, Value]:
    shell = _scripted_shell(commands, cmd_log=cmd_log_ir)
    return completed_bindings(run_inline_ir_with_shell(source, shell))


def evaluate_ir_raises_with_shell(
    source: str, commands: dict[str, ProcessCaptureResult]
) -> RunError:
    return uncaught_error(run_inline_ir_with_shell(source, _scripted_shell(commands)))
