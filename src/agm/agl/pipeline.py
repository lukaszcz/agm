"""PipelineDriver — top-of-stack orchestrator for the AgL execution pipeline.

Drives the full ``parse → scope → typecheck → matchcompile → lower/link → IR eval`` pipeline:
registers agents/codecs, validates host params, materializes output
contracts, and executes the program (or stops after static checking for
``agm exec --dry-run``).  Structured outputs use the JSON codec with
lenient-by-default recovery.

``agm.agl.runtime`` is the eval-free services layer (agents, codecs, params,
types).  This module is the top-of-stack host façade that depends on both
``runtime`` services and ``agm.agl.eval``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.eval.ir_interpreter import IrInterpreter
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.params import _materialize_ir_contracts, _prepare_ir_params
from agm.agl.runtime.types import (
    AgentDeclInfo as AgentDeclInfo,
)
from agm.agl.runtime.types import (
    CallSiteInfo as CallSiteInfo,
)
from agm.agl.runtime.types import (
    HostEnvironment,
    ParamDeclInfo,
)
from agm.agl.self_validation import self_validation_enabled

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.capabilities import HostCapabilities
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.matchcompile import MatchCompiledProgram
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.scope.program import ResolvedProgram
    from agm.agl.scope.symbols import ModuleResolution
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.values import ExceptionValue, Value
    from agm.agl.syntax.nodes import AgentDecl as AgentDeclNode
    from agm.agl.typecheck.env import OutputContractSpec
    from agm.agl.typecheck.program import CheckedProgram

_ResultT = TypeVar("_ResultT")

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"ask", "exec", "ask-request"})


class ArtifactProvenanceError(Exception):
    """A cached compiler artifact does not belong to the prepared source it is
    handed back with.

    Raised by AgL's optional self-validation only.  The artifact seam is
    internal — a caller passes back an artifact this pipeline produced for a
    specific prepared source — so a mismatch is a host-wiring bug with no
    user-facing remedy, not a diagnostic about the program.
    """


@dataclass(frozen=True, slots=True)
class ParamDiscovery:
    """Result of ``PipelineDriver.discover_params``."""

    params: tuple[ParamDeclInfo, ...]
    program_name: str | None
    checked: "CheckedProgram | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]
    compiled: "MatchCompiledProgram | None" = None


@dataclass(frozen=True, slots=True)
class ParamPreflight:
    """Result of ``PipelineDriver.preflight_params``.

    ``result``
        The check-only run result: ``ok`` iff every param validated.
    ``executable``
        The lowered program the params were checked against, or ``None`` when a
        pass before lowering failed.  Hand it back to
        ``PipelineDriver.run_prepared`` as ``executable`` to execute it
        without lowering the program a second time.
    """

    result: "RunResult"
    executable: "ExecutableProgram | None"


@dataclass(frozen=True, slots=True)
class PreparedProgram:
    """Result of the load + scope phase of an AgL multi-module program.

    Produced by :meth:`PipelineDriver.prepare_program` and consumed by
    :meth:`PipelineDriver.run_prepared`.  Properties mirror
    :class:`PreparedProgram` but read from the entry module of the program.

    ``resolved``
        The fully loaded and scope-resolved module graph, or ``None`` when
        loading or scope resolution failed (in which case ``diagnostics``
        holds the error and ``run_prepared`` short-circuits).
    ``diagnostics``
        Error-severity load/scope diagnostics; empty on success.
    ``warnings``
        Non-fatal lex (TAB) and scope warnings; present even on failure.
    ``companion_paths``
        Each loaded module's Python companion path (``None`` when the module
        declares no extern), keyed by module id.  Empty when loading failed.
        Consumed by ``run_prepared`` to import and resolve every
        declared extern before evaluation.
    """

    source: str
    entry_path: "Path | None"
    roots: "RootSet"
    resolved: "ResolvedProgram | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]
    companion_paths: "dict[ModuleId, Path | None]" = field(default_factory=dict)

    @property
    def declared_agents(self) -> tuple[AgentDeclInfo, ...]:
        """Agent declarations from the entry module, sorted by line/col.

        Empty when load or scope failed (``resolved is None``).
        """
        if self.resolved is None:
            return ()
        infos = [
            AgentDeclInfo(
                name=decl.name,
                runner=decl.runner,
                line=decl.span.start_line,
                col=decl.span.start_col,
            )
            for decl in self.resolved.entry_agents.values()
        ]
        infos.sort(key=lambda info: (info.line, info.col))
        return tuple(infos)

    @property
    def program_name(self) -> str | None:
        """The declared program name from the entry module, or ``None``."""
        from agm.agl.modules.ids import ENTRY_ID

        if self.resolved is None:
            return None
        entry_mod = self.resolved.modules.get(ENTRY_ID)
        if entry_mod is None:
            return None
        return entry_mod.resolved.program_name


@dataclass(frozen=True, slots=True)
class RunError:
    """Structured representation of an uncaught AgL exception.

    ``type_name`` is the exception's declared type name (e.g. ``"AgentParseError"``).
    ``fields`` is a mapping from field names to JSON-shaped Python values.
    ``line`` is the 1-based source line of the raise site when known (design
    : source location is part of runtime error reporting); ``None`` when
    the span was not threaded through (e.g. arithmetic errors inside expressions).
    ``col`` is the 1-based source column of the raise site; ``None`` when unknown.
    """

    type_name: str
    fields: dict[str, object]
    line: int | None = None
    col: int | None = None

    def to_message(self, *, include_trace_id: bool = False) -> str:
        """Render the single-line ``AgL exception: ...`` report for this error.

        Format: ``AgL exception: <Type>[: <message>][: at line L[, col C]]``,
        with a trailing ``: trace_id=<id>`` when *include_trace_id* is set and a
        trace id is present.  Shared by ``agm exec`` (with the trace id, design
        ) and the REPL failure echo so the two never diverge.
        """
        parts: list[str] = [f"AgL exception: {self.type_name}"]
        message = self.fields.get("message")
        if isinstance(message, str) and message:
            parts.append(message)
        if self.line is not None:
            if self.col is not None:
                parts.append(f"at line {self.line}, col {self.col}")
            else:
                parts.append(f"at line {self.line}")
        if include_trace_id:
            trace_id = self.fields.get("trace_id")
            if isinstance(trace_id, str) and trace_id:
                parts.append(f"trace_id={trace_id}")
        return ": ".join(parts)


@dataclass(slots=True)
class RunResult:
    """Result of a ``PipelineDriver.run`` call.

    ``ok``
        ``True`` iff there are no error-severity ``diagnostics`` **and** no
        uncaught AgL exception.  ``warnings`` never affect ``ok``.
    ``diagnostics``
        Pre-execution FAILURES only: error-severity items from
        lex/parse/scope/typecheck/matchcompile/param-validation.  Each entry has a
        ``.message`` (str) and a ``.line`` (int, 1-based).  Warnings are a
        SEPARATE channel and NEVER appear here; on a successful run this list is
        empty.
    ``warnings``
        Advisory warning-severity diagnostics (e.g. an unused declared agent)
        surfaced on EVERY path — success, static failure, param-validation
        failure, and uncaught exception.  Same ``Diagnostic`` type as
        ``diagnostics`` but with ``.severity == "warning"``.  Reported to the
        user but never cause the run to fail (never affect ``ok``).
    ``error``
        The uncaught AgL exception, or ``None``.  Set only when the program
        *started* executing but ended with an unhandled exception (exit code 2
        per the CLI contract).  ``None`` for pre-execution failures and for
        successful runs.
    ``bindings``
        Root-scope bindings after a successful run (name → Value).  Empty for
        failed runs.
    ``call_sites``
        Static call-site inventory populated when ``check_only=True``
        (``agm exec --dry-run``).  One entry per agent-call/exec site in
        source order.  Empty for ordinary runs.
    ``trace_path``
        Path of the JSONL trace file written during this run, or ``None``
        when logging was disabled (``--no-log``) or the run was a dry-run.
        This handle identifies the prepared program.
    """

    ok: bool
    diagnostics: list[Diagnostic]
    error: RunError | None
    warnings: list[Diagnostic] = field(default_factory=list)
    bindings: dict[str, Value] = field(default_factory=dict)
    call_sites: tuple[CallSiteInfo, ...] = field(default_factory=tuple)
    trace_path: Path | None = field(default=None)


class PipelineDriver:
    """Host API for the AgL interpreter.

    Constructor parameters
    ----------------------
    default_strict_json : bool
        When ``True`` the JSON codec defaults to strict parsing (only a bare
        JSON value with surrounding whitespace is accepted).  The default
        ``False`` enables lenient JSON recovery.
    default_loop_limit : int or None
        The host's global ``max-iters`` safety valve for unguarded loops
        (``while``/``do…until`` with no ``[n]`` bound and no ``for`` clause).
        ``None`` (the default) leaves the valve off; an integer caps unguarded
        loops at that many iterations, raising ``MaxIterationsExceeded``. Self-bounded
        loops (``for``, ``do[n]``) are never affected by this valve.  Resolved
        by the caller as ``--max-iters`` > ``[exec] max-iters``.
    default_agent : callable or None
        The callable used for the built-in ``ask`` agent.  ``None`` means
        no default agent is configured (only explicitly registered agents will
        be available).
    shell_exec_timeout : float or None
        Idle timeout (in seconds) applied to every ``exec`` shell call. ``None``
        means no timeout (the shell command may run indefinitely). This is the
        ``[exec] timeout`` config value, threaded in from the CLI.
    default_call_depth_limit : int or None
        Maximum call depth for recursive functions.  Exceeding
        this limit raises a ``RecursionError`` in the AgL program.  ``None``
        applies the canonical default (``IrInterpreter.DEFAULT_MAX_CALL_DEPTH``).
        Resolved by the caller as ``--max-call-depth`` > ``[exec] max-call-depth``.
    extern_registry : ExternRegistry or None
        Optional shared Python FFI registry. Hosts that run several drivers
        across one program invocation pass the same registry to each so
        companion module imports and module state are shared.
    """

    def __init__(
        self,
        *,
        default_strict_json: bool = False,
        default_loop_limit: int | None = None,
        default_agent: AgentFn | None = None,
        shell_exec_timeout: float | None = None,
        default_call_depth_limit: int | None = None,
        extern_registry: "ExternRegistry | None" = None,
    ) -> None:
        self._default_strict_json = default_strict_json
        self._default_loop_limit = default_loop_limit
        self._default_agent = default_agent
        self._shell_exec_timeout = shell_exec_timeout
        self._default_call_depth_limit = (
            default_call_depth_limit
            if default_call_depth_limit is not None
            else IrInterpreter.DEFAULT_MAX_CALL_DEPTH
        )
        self._agents: dict[str, AgentFn] = {}
        self._extern_registry = extern_registry
        # Extra codecs registered by the host (beyond the built-ins).
        self._extra_codecs: dict[str, "OutputCodec"] = {}
        # Cached assembled environment: invariant between registrations, so the
        # REPL's per-entry ``host_environment()`` calls reuse one bundle.  Any
        # ``register_*`` invalidates it.
        self._host_env_cache: HostEnvironment | None = None

    def register_agent(self, name: str, fn: AgentFn) -> None:
        """Register a named agent callable.

        Raises ``ValueError`` if ``name`` is a reserved name (``ask`` or
        ``exec``) or if an agent with that name has already been registered.
        """
        if name in _RESERVED_AGENT_NAMES:
            raise ValueError(
                f"Cannot register agent with reserved name {name!r}. "
                f"Reserved names: {sorted(_RESERVED_AGENT_NAMES)}"
            )
        if name in self._agents:
            raise ValueError(
                f"An agent named {name!r} is already registered. "
                "Duplicate registrations are not allowed."
            )
        self._agents[name] = fn
        self._host_env_cache = None

    def register_codec(self, codec: "OutputCodec") -> None:
        """Register a custom output codec.

        The codec's ``name`` property is the registration key.  The built-in
        codec names (``"text"`` and ``"json"``) are reserved and cannot be
        overridden.  Duplicate registrations (same name, regardless of
        implementation) are rejected.

        The codec must expose ``supported_kinds: frozenset[str]``; those kinds
        are surfaced in ``HostCapabilities.codec_kinds`` so the type-checker
        can validate ``format`` options at a call site.

        Raises ``ValueError`` for reserved or duplicate names.
        """
        from agm.agl.runtime.codec import BUILTIN_CODEC_NAMES

        name = codec.name
        if name in BUILTIN_CODEC_NAMES:
            raise ValueError(
                f"Cannot register codec with reserved name {name!r}. "
                f"Reserved codec names: {sorted(BUILTIN_CODEC_NAMES)}"
            )
        if name in self._extra_codecs:
            raise ValueError(
                f"A codec named {name!r} is already registered. "
                "Duplicate codec registrations are not allowed."
            )
        self._extra_codecs[name] = codec
        self._host_env_cache = None

    def host_environment(self) -> HostEnvironment:
        """Assemble the shared host environment from this runtime's registrations.

        Returns the ``AgentRegistry``, derived ``HostCapabilities``, and merged
        codec/renderer tables — the same bundle ``run`` builds internally.  An
        embedding host (e.g. ``ReplSession``) calls this to wire identical
        agent/codec/renderer backing without re-running the assembly itself.

        The bundle is invariant between registrations, so it is assembled once
        and cached; a ``register_*`` call invalidates the cache.  This spares the
        REPL re-assembling the whole environment on every entry / introspection.
        """
        if self._host_env_cache is not None:
            return self._host_env_cache
        self._host_env_cache = assemble_host_environment(
            agents=self._agents,
            default_agent=self._default_agent,
            extra_codecs=self._extra_codecs,
            extern_registry=self._extern_registry,
        )
        return self._host_env_cache

    def _execute_ir(
        self,
        executable: "ExecutableProgram",
        *,
        host_env: HostEnvironment,
        param_values: Mapping[str, object],
        check_only: bool,
        log_file: "Path | None",
        warnings: list[Diagnostic],
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
    ) -> RunResult:
        """Run a freshly lowered ``executable`` — the shared tail of the
        shared pipeline tail.

        Validates external params, materializes host codec contracts, honours
        the ``check_only`` dry-run stop (call-site inventory, no execution),
        then builds and runs the :class:`IrInterpreter`, mapping an uncaught
        ``AglRaise`` to a failing ``RunResult``.  All return paths carry
        *warnings*.
        """
        registry = host_env.registry

        ir_param_values, param_errors = _prepare_ir_params(executable, param_values)
        if param_errors:
            return RunResult(ok=False, diagnostics=param_errors, error=None, warnings=warnings)

        host_contracts, contract_errors = _materialize_ir_contracts(executable, host_env.codecs)
        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [check_only] --dry-run stop: the full static pipeline, param
        # validation, and contract materialization have all succeeded.  Stop
        # before executing any statement — no program output, no evaluation
        # side effects, no extern companion imports, and no trace is written.
        # ----------------------------------------------------------------
        if check_only:
            inventory = _build_call_inventory_from_ir(executable.dry_run_inventory)
            return RunResult(
                ok=True,
                diagnostics=[],
                error=None,
                warnings=list(warnings),
                bindings={},
                call_sites=tuple(inventory),
                trace_path=None,
            )

        # ----------------------------------------------------------------
        # Build and run the interpreter
        # ----------------------------------------------------------------
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise

        # Create the trace store for this run.  When log_file is None the
        # store is a no-op and no file is touched.
        trace = TraceStore(path=log_file)
        if log_file is not None:
            from agm.core.fs import mkdir

            try:
                mkdir(log_file.parent, parents=True, exist_ok=True)
            except OSError as exc:
                trace.disable(exc)
        trace.run_start()

        if host_settings_policy is not None:
            from agm.agl.runtime.host_settings import HostSettingsReconfigurer

            reconfigurer: HostSettingsReconfigurer | None = HostSettingsReconfigurer(
                registry=registry, trace=trace, policy=host_settings_policy
            )
        else:
            reconfigurer = None

        interp = IrInterpreter(
            executable,
            registry=registry,
            strict_json=self._default_strict_json,
            loop_limit=self._default_loop_limit,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            max_call_depth=self._default_call_depth_limit,
            param_values=ir_param_values,
            host_contracts=host_contracts,
            extern_registry=host_env.extern_registry,
            host_reconfigurer=reconfigurer,
            builtin_host_settings=builtin_host_settings,
        )

        try:
            root_bindings = interp.run()
        except AglRaise as exc:
            # Uncaught AgL exception (exit code 2 per the CLI contract).
            # ONLY the AgL exception carrier is caught here: an unexpected Python
            # exception is an interpreter bug and must propagate (crash loudly)
            # rather than masquerade as a user-facing pre-execution diagnostic.
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            # Record the uncaught exception in the trace.
            trace_id = str(error.fields.get("trace_id", ""))
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
                trace_id=trace_id,
                span=exc.span,
            )
            trace.run_end(ok=False)
            return RunResult(
                ok=False,
                diagnostics=[],
                error=error,
                warnings=list(warnings),
                bindings={},
                trace_path=trace.path,
            )

        trace.run_end(ok=True)

        return RunResult(
            ok=True,
            diagnostics=[],
            error=None,
            warnings=list(warnings),
            bindings=root_bindings,
            trace_path=trace.path,
        )

    @staticmethod
    def prepare_program(
        entry_source: str,
        *,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        default_stdlib: bool = True,
    ) -> PreparedProgram:
        """Load and resolve the program rooted at *entry_source* once.

        Drives ``parse → load imports → resolve_program`` for the entry module
        and every reachable module.

        Non-raising: any load (``ModuleNotFound``, ``AmbiguousModule``,
        ``ModulePrefixNotFound``, ``ImportEntryError``), parse
        (``AglSyntaxError``), or scope (``AglScopeError``) failure is
        captured into :attr:`PreparedProgram.diagnostics` rather than raised,
        with ``resolved`` left ``None``.  TAB advisories are captured
        as warnings via the lex-pass context manager.
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            MissingExternCompanion,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.modules.loader import load_graph
        from agm.agl.parser import AglSyntaxError
        from agm.agl.scope import AglScopeError
        from agm.agl.scope.program import resolve_program

        if roots is None:
            from pathlib import Path

            from agm.agl.modules.roots import assemble_roots
            from agm.config.module_roots import (
                ModuleRootsConfig,
                resolve_lib_root,
                resolve_stdlib_root,
            )

            cwd = Path.cwd()
            roots = assemble_roots(
                invocation_root=entry_path.resolve().parent if entry_path is not None else cwd,
                stdlib_root=resolve_stdlib_root(home=Path.home()),
                lib_root=resolve_lib_root(
                    ModuleRootsConfig(lib_root=None, extra=()), home=Path.home()
                ),
                configured=(),
                cli=(),
                cwd=cwd,
            )

        with tab_warning_collector() as tab_sink:
            try:
                graph = load_graph(
                    entry_source,
                    entry_path=entry_path,
                    roots=roots,
                    default_stdlib=default_stdlib,
                )
            except AglSyntaxError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    tuple(tab_sink),
                )
            except (
                ModuleNotFound,
                AmbiguousModule,
                ModulePrefixNotFound,
                ImportEntryError,
                MissingExternCompanion,
            ) as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    tuple(tab_sink),
                )
            except AglError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    tuple(tab_sink),
                )
            except Exception as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    tuple(tab_sink),
                )
        warnings: tuple[Diagnostic, ...] = tuple(tab_sink)

        try:
            resolved = resolve_program(graph)
        except AglScopeError as exc:
            return PreparedProgram(
                entry_source, entry_path, roots, None, (exc.to_diagnostic(),), warnings
            )
        except AglError as exc:
            return PreparedProgram(
                entry_source, entry_path, roots, None, (exc.to_diagnostic(),), warnings
            )
        except Exception as exc:
            return PreparedProgram(
                entry_source,
                entry_path,
                roots,
                None,
                (Diagnostic(message=f"Scope error: {exc}", line=1),),
                warnings,
            )

        # Collect scope warnings from the resolved program.
        all_warnings = (*warnings, *resolved.warnings)
        companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
        return PreparedProgram(
            entry_source, entry_path, roots, resolved, (), all_warnings, companion_paths
        )

    @staticmethod
    def declared_agents(
        source: str,
        *,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        default_stdlib: bool = True,
    ) -> tuple[AgentDeclInfo, ...]:
        """Return entry-module agent declarations without raising."""
        return PipelineDriver.prepare_program(
            source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
        ).declared_agents

    def run(
        self,
        source: str,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        default_stdlib: bool = True,
    ) -> RunResult:
        """Compile and run a program with the standard module roots by default."""
        return self.run_prepared(
            self.prepare_program(
                source, entry_path=entry_path, roots=roots, default_stdlib=default_stdlib
            ),
            param_values=param_values,
            check_only=check_only,
            log_file=log_file,
        )

    def discover_params(
        self,
        prepared: PreparedProgram,
        *,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> ParamDiscovery:
        """Discover typed ``param`` declarations from a resolved program.

        Runs typechecking and match compilation, then reads the entry module. A supplied artifact is
        reused; otherwise the successful artifact is returned for later
        lowering by :meth:`run_prepared`.
        """
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.syntax.nodes import ParamDecl

        if prepared.resolved is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=prepared.diagnostics,
                warnings=prepared.warnings,
            )

        capabilities = self.host_environment().capabilities
        if compiled is not None:
            if self_validation_enabled():
                _check_program_artifact_provenance(prepared.resolved, compiled.checked)
            if compiled.capabilities != capabilities:
                compiled = None

        if compiled is None:
            checked, tc_diagnostics = _run_typecheck_program(prepared.resolved, capabilities)
        else:
            checked = compiled.checked
            tc_diagnostics = ()
        all_warnings_list: list[Diagnostic] = list(prepared.warnings)
        if checked is not None:
            _append_checker_warnings(all_warnings_list, checked)
        all_warnings = tuple(all_warnings_list)

        if checked is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=tc_diagnostics,
                warnings=all_warnings,
            )

        if compiled is None:
            compiled, match_diagnostics = _run_matchcompile_program(checked)
            if compiled is None:
                return ParamDiscovery(
                    params=(),
                    program_name=prepared.program_name,
                    checked=checked,
                    diagnostics=match_diagnostics,
                    warnings=all_warnings,
                )

        entry_cm = checked.modules.get(ENTRY_ID)
        if entry_cm is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=(Diagnostic(message="Entry module not found in program", line=1),),
                warnings=all_warnings,
            )

        infos: list[ParamDeclInfo] = []
        for item in entry_cm.resolved.program.body.items:
            if isinstance(item, ParamDecl):
                param_type = entry_cm.type_env.get_binding_type(item.node_id)
                assert param_type is not None, (
                    f"Param {item.name!r} has no recorded binding type; checker invariant violated."
                )
                infos.append(
                    ParamDeclInfo(
                        name=item.name,
                        type=param_type,
                        has_default=item.default is not None,
                        line=item.span.start_line,
                        col=item.span.start_col,
                    )
                )
        infos.sort(key=lambda info: (info.line, info.col))
        return ParamDiscovery(
            params=tuple(infos),
            program_name=prepared.program_name,
            checked=checked,
            diagnostics=(),
            warnings=all_warnings,
            compiled=compiled,
        )

    def _wire_externs_or_fail(
        self,
        *,
        checked: "CheckedProgram",
        capabilities: "HostCapabilities",
        host_env: HostEnvironment,
        prepared: PreparedProgram,
        on_failure: "Callable[[list[Diagnostic]], _ResultT]",
    ) -> "_ResultT | None":
        """Import and resolve every extern companion, or build a failure result.

        Returns ``None`` when wiring succeeds, otherwise ``on_failure`` applied
        to the collected import/resolution diagnostics, so the caller decides
        which result dataclass carries them.
        """
        extern_diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=capabilities,
            registry=host_env.extern_registry,
            companion_paths=prepared.companion_paths,
        )
        if extern_diagnostics:
            return on_failure(extern_diagnostics)
        return None

    def run_prepared(
        self,
        prepared: PreparedProgram,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled: "MatchCompiledProgram | None" = None,
        checked: "CheckedProgram | None" = None,
        executable: "ExecutableProgram | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
    ) -> RunResult:
        """Execute an already loaded and scoped program without reloading.

        Resumes the pipeline at type checking: ``check_program`` → match
        compilation → ``lower_program`` → ``IrInterpreter``. Agents are
        entry-program-owned.

        When *prepared* carries a load/scope failure (``resolved is
        None``), its diagnostics are surfaced unchanged and nothing executes.

        ``compiled``
            When the caller has already typechecked and match-compiled the program
            (for example via :meth:`discover_params`), pass the result
            here to skip those static passes. ``None`` runs them here.

        ``executable``
            When the caller has already lowered this exact program (via
            :meth:`preflight_params`), pass the executable here to
            run it as-is: contract materialization and lowering are skipped, so
            a program is lowered only once however many times a host resumes it.
            ``None`` lowers here, before the check-only stop or evaluation.
        """
        result, _executable = self._run_program(
            prepared,
            param_values=param_values,
            check_only=check_only,
            log_file=log_file,
            compiled=compiled,
            checked=checked,
            executable=executable,
            host_settings_policy=host_settings_policy,
            builtin_host_settings=builtin_host_settings,
        )
        return result

    def preflight_params(
        self,
        prepared: PreparedProgram,
        *,
        param_values: Mapping[str, object] | None = None,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> ParamPreflight:
        """Validate external params against the program without executing it.

        Params are validated against the LOWERED program, so a host that must
        reject bad params before it commits to any run side effect has to lower
        first. This runs the static pipeline exactly as :meth:`run_prepared`
        does under ``check_only`` and hands the lowered program back, so the host
        can then execute it (``run_prepared(..., executable=...)``) without
        paying for a second lowering.
        """
        result, executable = self._run_program(
            prepared,
            param_values=param_values,
            check_only=True,
            compiled=compiled,
        )
        return ParamPreflight(result=result, executable=executable)

    def _run_program(
        self,
        prepared: PreparedProgram,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled: "MatchCompiledProgram | None" = None,
        checked: "CheckedProgram | None" = None,
        executable: "ExecutableProgram | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
    ) -> "tuple[RunResult, ExecutableProgram | None]":
        """Back program execution and parameter preflight with one pipeline body.

        Returns the run result together with the lowered program it ran (the one
        supplied as *executable*, or the one lowered here), or ``None`` when a
        pass before lowering failed.
        """
        if param_values is None:
            param_values = {}

        warnings: list[Diagnostic] = list(prepared.warnings)

        if prepared.resolved is None:
            return (
                RunResult(
                    ok=False,
                    diagnostics=list(prepared.diagnostics),
                    error=None,
                    warnings=warnings,
                ),
                None,
            )
        resolved = prepared.resolved

        host_env = self.host_environment()
        capabilities = host_env.capabilities
        if compiled is not None:
            if self_validation_enabled():
                _check_program_artifact_provenance(resolved, compiled.checked)
            if compiled.capabilities != capabilities:
                compiled = None
        if checked is not None and compiled is None:
            if self_validation_enabled():
                _check_program_artifact_provenance(resolved, checked)
            if checked.capabilities != capabilities:
                checked = None

        registry = host_env.registry

        # Agent reconciliation against entry module's declared agents.
        reconciliation_errors = _reconcile_agents(registry, resolved.entry_agents)
        if reconciliation_errors:
            return (
                RunResult(
                    ok=False,
                    diagnostics=reconciliation_errors,
                    error=None,
                    warnings=list(warnings),
                ),
                None,
            )

        # Reuse a supplied match-compiled program rather than repeating its
        # typecheck and match-compilation passes.
        tc_diagnostics: tuple[Diagnostic, ...]
        if compiled is not None:
            tc_diagnostics = ()
            checked = compiled.checked
        elif checked is None:
            checked, tc_diagnostics = _run_typecheck_program(resolved, capabilities)
        else:
            tc_diagnostics = ()
        if checked is None:
            return (
                RunResult(
                    ok=False,
                    diagnostics=list(tc_diagnostics),
                    error=None,
                    warnings=warnings,
                ),
                None,
            )

        _append_checker_warnings(warnings, checked)

        if compiled is None:
            compiled, match_diagnostics = _run_matchcompile_program(checked)
            if compiled is None:
                return (
                    RunResult(
                        ok=False,
                        diagnostics=list(match_diagnostics),
                        error=None,
                        warnings=warnings,
                    ),
                    None,
                )

        # An already lowered program carries its materialized contracts, so both
        # steps are skipped for it: the program is lowered exactly once per host
        # invocation, however many times the host resumes the pipeline.
        contract_payloads: "Mapping[int, ContractPayload]" = {}
        if executable is None:
            contract_payloads, contract_errors = _materialize_program_custom_contract_payloads(
                checked,
                host_env.codecs,
            )
            if contract_errors:
                return (
                    RunResult(
                        ok=False,
                        diagnostics=contract_errors,
                        error=None,
                        warnings=list(warnings),
                    ),
                    None,
                )

        if not check_only:
            # Extern (Python FFI) companions: import and resolve every declared
            # extern up front, gated by capability — fail-fast, before evaluation,
            # and after every static pass (so a static error elsewhere is reported
            # instead, with no companion import side effect). Dry-run stops before
            # this host-side import step to preserve its no-side-effects contract.
            run_failure = self._wire_externs_or_fail(
                checked=checked,
                capabilities=capabilities,
                host_env=host_env,
                prepared=prepared,
                on_failure=lambda extern_diagnostics: RunResult(
                    ok=False,
                    diagnostics=extern_diagnostics,
                    error=None,
                    warnings=warnings,
                ),
            )
            if run_failure is not None:
                return run_failure, None

        if executable is None:
            from agm.agl.lower import lower_program

            executable = lower_program(
                compiled,
                contract_payloads=contract_payloads,
            )

        return (
            self._execute_ir(
                executable,
                host_env=host_env,
                param_values=param_values,
                check_only=check_only,
                log_file=log_file,
                warnings=warnings,
                host_settings_policy=host_settings_policy,
                builtin_host_settings=builtin_host_settings,
            ),
            executable,
        )

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json

    @property
    def default_loop_limit(self) -> int | None:
        """Default max-iters valve (``None`` means off) for unguarded loops."""
        return self._default_loop_limit

    @property
    def shell_exec_timeout(self) -> float | None:
        """Idle timeout in seconds for ``exec`` shell calls (``None`` = no timeout)."""
        return self._shell_exec_timeout

    @property
    def default_call_depth_limit(self) -> int:
        """Maximum call depth for recursive functions."""
        return self._default_call_depth_limit

    def update_defaults(
        self,
        *,
        strict_json: bool,
        loop_limit: int | None,
        shell_exec_timeout: float | None,
    ) -> None:
        """Update the live engine defaults in place without losing registrations.

        Called by ``ReplSession`` after a successful entry, to carry that entry's
        engine settings — which a ``std.config`` write may have changed mid-entry
        — into the entries that follow.  Agent/codec registrations and the
        call-depth limit are preserved: only the three eval-consumed settings are
        updated.
        """
        self._default_strict_json = strict_json
        self._default_loop_limit = loop_limit
        self._shell_exec_timeout = shell_exec_timeout

    def reset_extern_registry(self) -> None:
        """Replace the cached extern registry with a fresh, empty one.

        Called by ``ReplSession.reset()`` so a session's extern state is
        discarded like every other session-scoped binding: after a reset, a
        library module's companion resolves and imports again as though the
        session were new. Agent/codec registrations and the rest of the
        assembled host environment are left untouched — only the extern
        registry is replaced. A no-op before the environment has ever been
        assembled (nothing cached yet to replace).
        """
        if self._host_env_cache is not None:
            from dataclasses import replace

            from agm.agl.runtime.externs import ExternRegistry

            self._extern_registry = ExternRegistry()
            self._host_env_cache = replace(
                self._host_env_cache, extern_registry=self._extern_registry
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_checker_warnings(
    warnings: list[Diagnostic],
    checked: "CheckedProgram",
) -> None:
    """Append one checked artifact's warnings at the typecheck phase boundary."""
    warnings.extend(checked.warnings)


def _check_artifact_provenance(
    *,
    prepared_entry: "ModuleId",
    prepared_modules: "Mapping[ModuleId, ModuleResolution]",
    compiled_entry: "ModuleId",
    compiled_modules: "Mapping[ModuleId, ModuleResolution]",
) -> None:
    """Assert a cached artifact wraps the exact prepared resolutions.

    Object identity is the provenance contract: structurally equal programs
    prepared in separate passes are not interchangeable compiler inputs.  Call
    sites guard this check with :func:`self_validation_enabled` — building its
    module mappings costs more than the production path should ever pay for an
    invariant it cannot violate.
    """
    same_provenance = (
        prepared_entry == compiled_entry
        and prepared_modules.keys() == compiled_modules.keys()
        and all(
            compiled_modules[module_id] is prepared_resolved
            for module_id, prepared_resolved in prepared_modules.items()
        )
    )
    if not same_provenance:
        raise ArtifactProvenanceError(
            "Cached match-compilation artifact does not belong to the prepared source."
        )


def _check_program_artifact_provenance(
    resolved: "ResolvedProgram",
    checked: "CheckedProgram",
) -> None:
    """Adapt whole-program artifacts to the shared provenance self-check.

    Match-compiled programs are checked through their `checked`, which
    carries the resolutions the compiler consumed.
    """
    _check_artifact_provenance(
        prepared_entry=resolved.entry_id,
        prepared_modules={
            module_id: module.resolved for module_id, module in resolved.modules.items()
        },
        compiled_entry=checked.entry_id,
        compiled_modules={
            module_id: module.resolved for module_id, module in checked.modules.items()
        },
    )


def _run_typecheck_program(
    resolved: "ResolvedProgram",
    capabilities: "HostCapabilities",
) -> "tuple[CheckedProgram | None, tuple[Diagnostic, ...]]":
    """Run the program typecheck pass without raising."""
    from agm.agl.typecheck.program import check_program

    try:
        return check_program(resolved, capabilities), ()
    except AglError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        return None, (Diagnostic(message=f"Type error: {exc}", line=1),)


def _run_matchcompile_program(
    checked: "CheckedProgram",
) -> "tuple[MatchCompiledProgram | None, tuple[Diagnostic, ...]]":
    """Run program-level match compilation without raising."""
    from agm.agl.matchcompile import (
        MatchCompiledProgram,
        compile_program_matches,
        diagnostics_from_match_issues,
    )

    try:
        result = compile_program_matches(checked)
        if result.compiled is None:
            return None, diagnostics_from_match_issues(result.issues)
        if not isinstance(result.compiled, MatchCompiledProgram):
            raise TypeError("program match compilation returned a module artifact")
        return result.compiled, ()
    except Exception as exc:
        return None, (Diagnostic(message=f"Match compilation error: {exc}", line=1),)


def _materialize_custom_contract_payloads(
    specs: "Mapping[int, OutputContractSpec]",
    codecs: "Mapping[str, OutputCodec]",
    type_table: "TypeTable",
) -> tuple[dict[int, "ContractPayload"], list[Diagnostic]]:
    """Run custom codec contract hooks before lowering and keep only typeless data."""
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.runtime.codec import BUILTIN_CODEC_NAMES
    from agm.agl.runtime.contract import materialize_contract

    payloads: dict[int, ContractPayload] = {}
    errors: list[Diagnostic] = []
    for node_id, spec in specs.items():
        if spec.codec_name in BUILTIN_CODEC_NAMES:
            continue
        try:
            contract = materialize_contract(spec, codecs, type_table)
            json_schema = (
                None
                if contract.json_schema is None
                else json.dumps(contract.json_schema, sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            errors.append(Diagnostic(message=f"Contract error: {exc}", line=1))
            continue
        payloads[node_id] = ContractPayload(
            json_schema=json_schema,
            decode=contract.decode,
            format_instructions=contract.format_instructions,
            defs=contract.defs,
        )
    return payloads, errors


def _materialize_program_custom_contract_payloads(
    checked: "CheckedProgram",
    codecs: "Mapping[str, OutputCodec]",
) -> tuple[dict[int, "ContractPayload"], list[Diagnostic]]:
    """Materialize custom contract payloads for every module in a checked program."""
    payloads: dict[int, "ContractPayload"] = {}
    errors: list[Diagnostic] = []
    for checked_module in checked.modules.values():
        module_payloads, module_errors = _materialize_custom_contract_payloads(
            checked_module.contract_specs,
            codecs,
            checked_module.type_env.type_table,
        )
        payloads.update(module_payloads)
        errors.extend(module_errors)
    return payloads, errors


def _extern_declaration_sort_key(pair: "tuple[ModuleId, str]") -> "tuple[tuple[str, ...], str]":
    """Sort key for deterministic ``(module_id, name)`` diagnostic ordering."""
    return (pair[0].segments, pair[1])


def _extern_declarations(
    checked: "CheckedProgram",
) -> list[tuple["ModuleId", str]]:
    """Return ``(module_id, name)`` for every ``extern def`` in *checked*.

    Sorted for deterministic diagnostic ordering; the checked AST (not the
    IR) is the source of truth here since the IR has no extern table yet.
    """
    declarations: list[tuple[ModuleId, str]] = [
        (mid, name)
        for mid, mod in checked.modules.items()
        for name, funcdef in mod.resolved.declared_functions.items()
        if funcdef.is_extern
    ]
    declarations.sort(key=_extern_declaration_sort_key)
    return declarations


def _wire_extern_registry(
    *,
    checked: "CheckedProgram",
    capabilities: "HostCapabilities",
    registry: "ExternRegistry",
    companion_paths: "Mapping[ModuleId, Path | None]",
) -> list[Diagnostic]:
    """Import every companion and resolve every declared extern, up front.

    Returns diagnostics — a single capability-gate diagnostic when the host
    disables ``supports_extern`` and the program declares any extern, or one
    diagnostic per companion that fails to import or resolve — collected
    before any evaluation.  Returns ``[]`` immediately when the program
    declares no extern, regardless of the capability (non-extern programs are
    never affected).  Mutates *registry* in place, so a ``PipelineDriver``
    that reuses the same ``HostEnvironment`` across multiple runs (e.g. the
    REPL) imports each companion only once.
    """
    from agm.agl.runtime.externs import ExternImportError, ExternResolutionError

    # A companion path is recorded (non-``None``) exactly for extern-declaring
    # modules, so this cheap check short-circuits the common no-extern program
    # before the full declared-function walk in ``_extern_declarations`` — and,
    # equivalently, gates the capability diagnostic without that walk.
    if not any(path is not None for path in companion_paths.values()):
        return []
    if not capabilities.supports_extern:
        return [
            Diagnostic(
                message=(
                    "program declares one or more extern definitions, but this "
                    "host does not support the Python FFI (supports_extern is "
                    "disabled)"
                ),
                line=1,
            )
        ]
    declarations = _extern_declarations(checked)

    diagnostics: list[Diagnostic] = []
    loaded_modules: set["ModuleId"] = set()
    failed_modules: set["ModuleId"] = set()
    for mid, name in declarations:
        if mid in failed_modules:
            # This module's companion already failed to import; every extern
            # it declares was already reported by that one diagnostic.
            continue
        if mid not in loaded_modules:
            companion_path = companion_paths.get(mid)
            assert companion_path is not None, (
                f"module {mid.display()!r} declares extern {name!r} but has no "
                "companion path recorded by the loader"
            )
            try:
                registry.load_companion(mid, companion_path)
            except ExternImportError as exc:
                diagnostics.append(exc.to_diagnostic())
                failed_modules.add(mid)
                continue
            loaded_modules.add(mid)
        try:
            registry.resolve(mid, name)
        except ExternResolutionError as exc:
            diagnostics.append(exc.to_diagnostic())
    return diagnostics


def _reconcile_agents(
    registry: "AgentRegistry",
    declared_agents: "Mapping[str, AgentDeclNode]",
) -> list[Diagnostic]:
    """Enforce the source↔host agent contract.

    Returns error :class:`Diagnostic`s for BOTH contract violations (never
    stopping at the first) so the user sees every mismatch at once:

    - **Registered-but-undeclared** (decision 1): a name in
      ``registry.agent_names`` that the source never declares.  Reported at
      ``line=1`` (a registration has no source span).
    - **Declared-but-unbacked** (decision 11): a declared agent with no
      dedicated registration AND no default agent.  Reported at the
      declaration's ``span.start_line``.  When a default agent IS present every
      declared name is backed by it, so this never fires.

    Order is deterministic: registered-but-undeclared first (sorted by name),
    then declared-but-unbacked (sorted by name).  ``declared_agents`` maps a
    declared name to its ``AgentDecl`` (only ``.span.start_line`` is read).
    """
    errors: list[Diagnostic] = []

    declared_names = set(declared_agents)
    for name in sorted(registry.agent_names - declared_names):
        errors.append(
            Diagnostic(
                message=(
                    f"Agent {name!r} is registered but never declared in the "
                    f"program. Declare it with `agent {name}` or remove the "
                    "registration."
                ),
                line=1,
            )
        )

    if not registry.has_default_agent:
        for name in sorted(declared_names - registry.agent_names):
            errors.append(
                Diagnostic(
                    message=(
                        f"Agent {name!r} is declared but has no backing: "
                        "register it with register_agent or configure a "
                        "default agent."
                    ),
                    line=declared_agents[name].span.start_line,
                    column=declared_agents[name].span.start_col,
                    end_line=declared_agents[name].span.end_line,
                    end_column=declared_agents[name].span.end_col,
                )
            )

    return errors


def assemble_host_environment(
    *,
    agents: dict[str, AgentFn],
    default_agent: AgentFn | None,
    extra_codecs: dict[str, "OutputCodec"],
    extern_registry: "ExternRegistry | None" = None,
) -> HostEnvironment:
    """Assemble the shared host runtime environment from registrations.

    Builds the merged codec table, the ``AgentRegistry``, and the derived
    ``HostCapabilities`` exactly as ``PipelineDriver.run`` does inline.
    Used by BOTH ``run`` and ``ReplSession`` so the two share identical
    agent/codec wiring (CARRY-IN 1: codec_kinds are derived from the actual
    registries, not from duplicated constants).
    """
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import JsonCodec, TextCodec
    from agm.agl.runtime.externs import ExternRegistry

    text_codec = TextCodec()
    json_codec = JsonCodec()

    # Merge built-in codecs with any host-registered extras.
    all_codecs: dict[str, "OutputCodec"] = {
        text_codec.name: text_codec,
        json_codec.name: json_codec,
        **extra_codecs,
    }

    registry = AgentRegistry(
        named=dict(agents),
        default_agent=default_agent,
    )
    capabilities = HostCapabilities(
        agent_names=registry.agent_names,
        has_default_agent=registry.has_default_agent,
        supports_shell_exec=True,
        supports_extern=True,
        codec_kinds={name: codec.supported_kinds for name, codec in all_codecs.items()},
    )
    return HostEnvironment(
        registry=registry,
        capabilities=capabilities,
        codecs=all_codecs,
        extern_registry=extern_registry if extern_registry is not None else ExternRegistry(),
    )


def exception_value_to_run_error(
    exc: "ExceptionValue",
    *,
    span: "object" = None,  # SourceSpan | None — avoids import cycle
) -> RunError:
    """Convert an ``ExceptionValue`` to a ``RunError`` for ``RunResult``.

    Field values are converted via the shared serializer, which preserves
    ``Decimal`` exactness (never routed through binary ``float``; design ).

    *span* is the optional raise-site source span threaded from ``AglRaise``;
    when present, ``RunError.line`` and ``RunError.col`` are populated from it
    so the CLI can include the source location in its exit-2 error output.
    """
    from agm.agl.ir.ids import Location
    from agm.agl.runtime.serialize import value_to_json_obj
    from agm.agl.syntax.spans import SourceSpan

    fields: dict[str, object] = {k: value_to_json_obj(v) for k, v in exc.fields.items()}
    line: int | None = None
    col: int | None = None
    if isinstance(span, (SourceSpan, Location)):
        line = span.start_line
        col = span.start_col
    return RunError(type_name=exc.display_name, fields=fields, line=line, col=col)


def _build_call_inventory_from_ir(entries: "tuple[object, ...]") -> list[CallSiteInfo]:
    """Convert lowering-owned dry-run metadata to the public runtime shape."""
    from agm.agl.ir.program import DryRunEntry

    return [
        CallSiteInfo(
            callee=entry.callee,
            target_type=entry.target_type_label,
            codec_name=entry.codec_name,
            has_schema=entry.has_schema,
            parse_policy=entry.parse_policy,
            line=entry.line,
            col=entry.col,
        )
        for entry in entries
        if isinstance(entry, DryRunEntry)
    ]
