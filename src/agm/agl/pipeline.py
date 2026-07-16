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

from agm.agl.diagnostics import AglError, Diagnostic, diagnostic_from_span
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

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.capabilities import HostCapabilities
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.matchcompile import MatchCompiledModuleGraph, MatchCompiledProgram
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.scope.graph import ResolvedModuleGraph
    from agm.agl.scope.symbols import ResolvedProgram
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.values import ExceptionValue, Value
    from agm.agl.syntax.nodes import AgentDecl as AgentDeclNode
    from agm.agl.syntax.nodes import Program
    from agm.agl.syntax.spans import SourceSpan
    from agm.agl.typecheck.env import CheckedProgram as CheckedProgramType
    from agm.agl.typecheck.env import OutputContractSpec
    from agm.agl.typecheck.graph import CheckedModuleGraph

_ResultT = TypeVar("_ResultT")

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"ask", "exec", "ask-request"})


@dataclass(frozen=True, slots=True)
class ParamDiscovery:
    """Result of ``PipelineDriver.discover_params``."""

    params: tuple[ParamDeclInfo, ...]
    program_name: str | None
    checked: "CheckedProgramType | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]
    checked_graph: "CheckedModuleGraph | None" = None
    compiled: "MatchCompiledProgram | None" = None
    compiled_graph: "MatchCompiledModuleGraph | None" = None


@dataclass(frozen=True, slots=True)
class PreparedProgram:
    """Result of the lex + parse + scope phase of an AgL program.

    Produced by :meth:`PipelineDriver.prepare` and consumed by
    :meth:`PipelineDriver.run_prepared`, so those two static phases run exactly
    ONCE even when a host inspects :attr:`declared_agents` (to wire registrations)
    before executing.  ``run(source)`` is exactly
    ``run_prepared(prepare(source))``.

    ``program`` / ``resolved``
        The parsed AST and resolved program, or ``None`` when parse / scope
        failed (in which case ``diagnostics`` holds the error and
        ``run_prepared`` short-circuits to an ``ok=False`` result).
    ``diagnostics``
        Error-severity parse/scope diagnostics; empty on success.
    ``warnings``
        Non-fatal lex (TAB) and scope warnings; present even on failure.
    """

    source: str
    program: "Program | None"
    resolved: "ResolvedProgram | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]

    @property
    def declared_agents(self) -> tuple[AgentDeclInfo, ...]:
        """The ``agent`` declarations in source, sorted by line/col.

        Empty when parse or scope failed (``resolved is None``).
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
            for decl in self.resolved.declared_agents.values()
        ]
        infos.sort(key=lambda info: (info.line, info.col))
        return tuple(infos)

    @property
    def program_name(self) -> str | None:
        """The declared program name, or ``None`` when undeclared."""
        if self.resolved is None:
            return None
        return self.resolved.program_name


@dataclass(frozen=True, slots=True)
class PreparedGraph:
    """Result of the load + scope phase of an AgL multi-module program.

    Produced by :meth:`PipelineDriver.prepare_program` and consumed by
    :meth:`PipelineDriver.run_prepared_graph`.  Properties mirror
    :class:`PreparedProgram` but read from the entry module of the graph.

    ``resolved_graph``
        The fully loaded and scope-resolved module graph, or ``None`` when
        loading or scope resolution failed (in which case ``diagnostics``
        holds the error and ``run_prepared_graph`` short-circuits).
    ``diagnostics``
        Error-severity load/scope diagnostics; empty on success.
    ``warnings``
        Non-fatal lex (TAB) and scope warnings; present even on failure.
    ``companion_paths``
        Each loaded module's Python companion path (``None`` when the module
        declares no extern), keyed by module id.  Empty when loading failed.
        Consumed by ``run_prepared_graph`` to import and resolve every
        declared extern before evaluation.
    """

    source: str
    entry_path: "Path | None"
    roots: "RootSet"
    resolved_graph: "ResolvedModuleGraph | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]
    companion_paths: "dict[ModuleId, Path | None]" = field(default_factory=dict)

    @property
    def declared_agents(self) -> tuple[AgentDeclInfo, ...]:
        """Agent declarations from the entry module, sorted by line/col.

        Empty when load or scope failed (``resolved_graph is None``).
        """
        if self.resolved_graph is None:
            return ()
        infos = [
            AgentDeclInfo(
                name=decl.name,
                runner=decl.runner,
                line=decl.span.start_line,
                col=decl.span.start_col,
            )
            for decl in self.resolved_graph.entry_agents.values()
        ]
        infos.sort(key=lambda info: (info.line, info.col))
        return tuple(infos)

    @property
    def program_name(self) -> str | None:
        """The declared program name from the entry module, or ``None``."""
        from agm.agl.modules.ids import ENTRY_ID

        if self.resolved_graph is None:
            return None
        entry_mod = self.resolved_graph.modules.get(ENTRY_ID)
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
        This handle identifies the prepared graph.
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
        ``None`` (the default) means the valve is OFF — unbounded loops run
        until they self-terminate.  An ``int`` caps unguarded loops at that
        many iterations, raising ``MaxIterationsExceeded``.  Self-bounded
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

    @staticmethod
    def prepare(source: str) -> PreparedProgram:
        """Lex + parse + scope *source* ONCE, capturing diagnostics and warnings.

        This is the single front-end phase shared by :meth:`declared_agents` and
        :meth:`run`: a host that needs the declared-agent inventory to wire
        registrations calls ``prepare`` once, reads
        :attr:`PreparedProgram.declared_agents`, then hands the SAME
        :class:`PreparedProgram` to :meth:`run_prepared` — so the source is never
        parsed or scoped twice.

        Side-effect-free and NON-raising: any parse (``AglSyntaxError``) or scope
        (``AglScopeError``) failure — and any unexpected error in those passes —
        is captured into :attr:`PreparedProgram.diagnostics` rather than raised,
        with ``program`` / ``resolved`` left ``None``.  TAB advisories come from
        the parse's single lex pass (no separate TAB scan).
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.parser import AglSyntaxError, parse_program
        from agm.agl.scope import AglScopeError, resolve

        # The collector captures the lexer's TAB advisories during the parse —
        # populated even when the parse fails (the scan completes before the
        # grammar is consulted), so they surface on every return path.
        with tab_warning_collector() as tab_sink:
            try:
                program = parse_program(source)
            except AglSyntaxError as exc:
                return PreparedProgram(source, None, None, (exc.to_diagnostic(),), tuple(tab_sink))
            except AglError as exc:
                return PreparedProgram(source, None, None, (exc.to_diagnostic(),), tuple(tab_sink))
            except Exception as exc:
                return PreparedProgram(
                    source,
                    None,
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    tuple(tab_sink),
                )
        warnings: tuple[Diagnostic, ...] = tuple(tab_sink)

        try:
            resolved = resolve(program)
        except AglScopeError as exc:
            return PreparedProgram(source, program, None, (exc.to_diagnostic(),), warnings)
        except AglError as exc:
            return PreparedProgram(source, program, None, (exc.to_diagnostic(),), warnings)
        except Exception as exc:
            return PreparedProgram(
                source,
                program,
                None,
                (Diagnostic(message=f"Scope error: {exc}", line=1),),
                warnings,
            )

        # Scope warnings (e.g. a declared-but-uncalled agent) join the lex ones.
        return PreparedProgram(source, program, resolved, (), (*warnings, *resolved.warnings))

    @staticmethod
    def declared_agents(source: str) -> tuple[AgentDeclInfo, ...]:
        """Return the agents declared in *source* (parse + scope only).

        Thin wrapper over :meth:`prepare`: returns one :class:`AgentDeclInfo` per
        ``agent`` declaration, sorted by source line/col.  NON-raising — on any
        parse or scope error it returns ``()`` (the subsequent ``run`` resurfaces
        the diagnostic).
        """
        return PipelineDriver.prepare(source).declared_agents

    def discover_params(self, prepared: PreparedProgram) -> ParamDiscovery:
        """Discover typed ``param`` declarations from a resolved program.

        Runs typechecking and match compilation after :meth:`prepare` has
        resolved the source. The successful artifact is returned for reuse by
        :meth:`run_prepared`, avoiding repeated static passes before lowering.
        """
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
        checked, tc_diagnostics = _run_typecheck(prepared.resolved, capabilities)
        all_warnings_list = list(prepared.warnings)
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

        compiled, match_diagnostics = _run_matchcompile(checked)
        if compiled is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=checked,
                diagnostics=match_diagnostics,
                warnings=all_warnings,
            )

        assert prepared.program is not None
        infos: list[ParamDeclInfo] = []
        for item in prepared.program.body.items:
            if isinstance(item, ParamDecl):
                param_type = checked.type_env.get_binding_type(item.node_id)
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

    def run(
        self,
        source: str,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
    ) -> RunResult:
        """Parse, analyse, and (unless ``check_only``) execute an AgL program.

        Pipeline:
            parse → resolve → typecheck (with HostCapabilities) →
            matchcompile → lower → validate params → materialize contracts → eval

        Convenience wrapper: ``run(source)`` is exactly
        ``run_prepared(prepare(source))``.  A host that needs the declared-agent
        inventory before execution should call :meth:`prepare` once and pass the
        result to :meth:`run_prepared`, so the source is parsed and scoped only
        once.

        When ``check_only`` is ``True`` (``agm exec --dry-run``) the runtime
        runs through match compilation and lowering, validates params, and
        materializes contracts, then STOPS before executing any statement: a
        clean program returns ``ok=True`` with no bindings and produces no
        program output; static/param errors still return ``ok=False``. On a
        clean ``check_only`` run, ``RunResult.call_sites`` contains the static
        call inventory printed by ``agm exec --dry-run``.

        ``log_file`` is the path of the JSONL trace file to write.  When
        ``None`` (the default) no trace is written.  Dry-run (``check_only``)
        never writes a trace regardless of *log_file*.

        Returns a ``RunResult`` capturing the outcome.
        """
        return self.run_prepared(
            self.prepare(source),
            param_values=param_values,
            check_only=check_only,
            log_file=log_file,
        )

    def run_prepared(
        self,
        prepared: PreparedProgram,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled: "MatchCompiledProgram | None" = None,
        checked: "CheckedProgramType | None" = None,
    ) -> RunResult:
        """Execute an already parsed + scoped program (no re-parsing).

        Resumes the pipeline at type checking: reconcile agents → typecheck →
        matchcompile → lower → validate params → materialize contracts →
        eval. A supplied ``compiled`` artifact reuses the exact prepared
        program's typecheck and match-compilation results. See :meth:`run` for
        the ``check_only`` / ``log_file`` semantics. When *prepared* carries a
        captured parse/scope failure (``resolved is None``), its diagnostics are
        surfaced unchanged and nothing executes.
        """
        if param_values is None:
            param_values = {}

        # Lex + scope warnings travel with the prepared program (present on
        # every return path, like typecheck warnings).
        warnings: list[Diagnostic] = list(prepared.warnings)

        # A parse/scope failure captured by ``prepare`` short-circuits here.
        if prepared.resolved is None:
            return RunResult(
                ok=False,
                diagnostics=list(prepared.diagnostics),
                error=None,
                warnings=warnings,
            )
        resolved = prepared.resolved
        program = prepared.program
        assert program is not None  # resolved set ⇒ parse succeeded
        source = prepared.source

        # ----------------------------------------------------------------
        # Build the host environment (registry + capabilities + codecs +
        # renderers) from registrations.  Shared with ``ReplSession`` so the
        # incremental driver wires identical agent/codec/renderer backing.
        # ----------------------------------------------------------------
        host_env = self.host_environment()
        registry = host_env.registry
        capabilities = host_env.capabilities

        from agm.agl.modules.ids import ENTRY_ID

        if compiled is not None:
            cache_diagnostic = _cached_artifact_provenance_diagnostic(
                prepared_entry=ENTRY_ID,
                prepared_modules={ENTRY_ID: resolved},
                compiled_entry=ENTRY_ID,
                compiled_modules={ENTRY_ID: compiled.checked.resolved},
                span=program.span,
            )
            if cache_diagnostic is not None:
                return RunResult(
                    ok=False,
                    diagnostics=[cache_diagnostic],
                    error=None,
                    warnings=warnings,
                )
            if compiled.capabilities != capabilities:
                compiled = None
        if checked is not None and compiled is None:
            cache_diagnostic = _cached_checked_artifact_provenance_diagnostic(
                prepared_entry=ENTRY_ID,
                prepared_modules={ENTRY_ID: resolved},
                checked_entry=ENTRY_ID,
                checked_modules={ENTRY_ID: checked.resolved},
                span=program.span,
            )
            if cache_diagnostic is not None:
                return RunResult(
                    ok=False,
                    diagnostics=[cache_diagnostic],
                    error=None,
                    warnings=warnings,
                )
            if checked.capabilities != capabilities:
                checked = None

        # ----------------------------------------------------------------
        # [2b] Source↔host agent reconciliation
        #
        # Enforce the source/host contract BEFORE execution (preferred over
        # waiting for typecheck — a broken contract should preempt execution).
        # Reported on the same channel as param-validation / host-config
        # errors: a non-empty diagnostics list ⇒ ok=False, nothing executes.
        # ----------------------------------------------------------------
        reconciliation_errors = _reconcile_agents(registry, resolved.declared_agents)
        if reconciliation_errors:
            return RunResult(
                ok=False,
                diagnostics=reconciliation_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [3] Type checking
        # ----------------------------------------------------------------
        if compiled is None and checked is None:
            checked, tc_diagnostics = _run_typecheck(resolved, capabilities)
            if checked is None:
                return RunResult(
                    ok=False,
                    diagnostics=list(tc_diagnostics),
                    error=None,
                    warnings=warnings,
                )
        elif compiled is not None:
            checked = compiled.checked

        assert checked is not None
        _append_checker_warnings(warnings, checked)

        if compiled is None:
            compiled, match_diagnostics = _run_matchcompile(checked)
            if compiled is None:
                return RunResult(
                    ok=False,
                    diagnostics=list(match_diagnostics),
                    error=None,
                    warnings=warnings,
                )

        contract_payloads, contract_errors = _materialize_custom_contract_payloads(
            checked.contract_specs,
            host_env.codecs,
            checked.type_env.type_table,
        )
        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        from agm.agl.lower import lower_program

        executable = lower_program(
            compiled,
            source_text=source,
            source_label="<entry>",
            validate=True,
            contract_payloads=contract_payloads,
        )

        return self._execute_ir(
            executable,
            host_env=host_env,
            param_values=param_values,
            check_only=check_only,
            log_file=log_file,
            warnings=warnings,
        )

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
        single-program and module-graph pipelines.

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

            mkdir(log_file.parent, parents=True, exist_ok=True)
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
                trace_path=log_file,
            )

        trace.run_end(ok=True)

        return RunResult(
            ok=True,
            diagnostics=[],
            error=None,
            warnings=list(warnings),
            bindings=root_bindings,
            trace_path=log_file,
        )

    @staticmethod
    def prepare_program(
        entry_source: str,
        *,
        entry_path: "Path | None",
        roots: "RootSet",
        default_stdlib: bool = True,
    ) -> PreparedGraph:
        """Load + scope the full module graph for *entry_source* ONCE.

        This is the graph-mode analogue of :meth:`prepare`.  Drives the load
        pipeline (``parse → BFS-load-imports → resolve_graph``) for a
        multi-file AgL program rooted at *entry_source*.

        Non-raising: any load (``ModuleNotFound``, ``AmbiguousModule``,
        ``ModulePrefixNotFound``, ``ImportEntryError``), parse
        (``AglSyntaxError``), or scope (``AglScopeError``) failure is
        captured into :attr:`PreparedGraph.diagnostics` rather than raised,
        with ``resolved_graph`` left ``None``.  TAB advisories are captured
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
        from agm.agl.scope.graph import resolve_graph

        with tab_warning_collector() as tab_sink:
            try:
                graph = load_graph(
                    entry_source,
                    entry_path=entry_path,
                    roots=roots,
                    default_stdlib=default_stdlib,
                )
            except AglSyntaxError as exc:
                return PreparedGraph(
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
                return PreparedGraph(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    tuple(tab_sink),
                )
            except AglError as exc:
                return PreparedGraph(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    tuple(tab_sink),
                )
            except Exception as exc:
                return PreparedGraph(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    tuple(tab_sink),
                )
        warnings: tuple[Diagnostic, ...] = tuple(tab_sink)

        try:
            resolved_graph = resolve_graph(graph)
        except AglScopeError as exc:
            return PreparedGraph(
                entry_source, entry_path, roots, None, (exc.to_diagnostic(),), warnings
            )
        except AglError as exc:
            return PreparedGraph(
                entry_source, entry_path, roots, None, (exc.to_diagnostic(),), warnings
            )
        except Exception as exc:
            return PreparedGraph(
                entry_source,
                entry_path,
                roots,
                None,
                (Diagnostic(message=f"Scope error: {exc}", line=1),),
                warnings,
            )

        # Collect scope warnings from the resolved graph.
        all_warnings = (*warnings, *resolved_graph.warnings)
        companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
        return PreparedGraph(
            entry_source, entry_path, roots, resolved_graph, (), all_warnings, companion_paths
        )

    def discover_params_graph(
        self,
        prepared: PreparedGraph,
        *,
        compiled_graph: "MatchCompiledModuleGraph | None" = None,
    ) -> ParamDiscovery:
        """Discover typed ``param`` declarations from a resolved module graph.

        Graph-mode analogue of :meth:`discover_params`. Runs typechecking and
        match compilation, then reads the entry module. A supplied artifact is
        reused; otherwise the successful artifact is returned for later
        lowering by :meth:`run_prepared_graph`.
        """
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.syntax.nodes import ParamDecl

        if prepared.resolved_graph is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=prepared.diagnostics,
                warnings=prepared.warnings,
                checked_graph=None,
            )

        capabilities = self.host_environment().capabilities
        if compiled_graph is not None:
            cache_diagnostic = _cached_graph_artifact_provenance_diagnostic(
                prepared.resolved_graph, compiled_graph
            )
            if cache_diagnostic is not None:
                return ParamDiscovery(
                    params=(),
                    program_name=prepared.program_name,
                    checked=None,
                    diagnostics=(cache_diagnostic,),
                    warnings=prepared.warnings,
                    checked_graph=None,
                )
            if compiled_graph.capabilities != capabilities:
                compiled_graph = None

        if compiled_graph is None:
            checked_graph, tc_diagnostics = _run_typecheck_graph(
                prepared.resolved_graph, capabilities
            )
        else:
            checked_graph = compiled_graph.checked_graph
            tc_diagnostics = ()
        all_warnings_list: list[Diagnostic] = list(prepared.warnings)
        if checked_graph is not None:
            _append_checker_warnings(all_warnings_list, checked_graph)
        all_warnings = tuple(all_warnings_list)

        if checked_graph is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=tc_diagnostics,
                warnings=all_warnings,
                checked_graph=None,
            )

        if compiled_graph is None:
            compiled_graph, match_diagnostics = _run_matchcompile_graph(checked_graph)
            if compiled_graph is None:
                return ParamDiscovery(
                    params=(),
                    program_name=prepared.program_name,
                    checked=None,
                    diagnostics=match_diagnostics,
                    warnings=all_warnings,
                    checked_graph=checked_graph,
                )

        entry_cm = checked_graph.modules.get(ENTRY_ID)
        if entry_cm is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=(Diagnostic(message="Entry module not found in graph", line=1),),
                warnings=all_warnings,
            )

        # Build a stub CheckedProgram for compatibility with existing param wiring.
        from agm.agl.typecheck.env import CheckedProgram as _CP

        stub_checked = _CP(
            resolved=entry_cm.resolved,
            node_types=entry_cm.node_types,
            contract_specs=entry_cm.contract_specs,
            call_sites=entry_cm.call_sites,
            warnings=entry_cm.warnings,
            type_env=entry_cm.type_env,
            function_signatures=entry_cm.function_signatures,
            cast_specs=entry_cm.cast_specs,
            argument_bindings=entry_cm.argument_bindings,
            partial_calls=entry_cm.partial_calls,
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
            checked=stub_checked,
            diagnostics=(),
            warnings=all_warnings,
            checked_graph=checked_graph,
            compiled_graph=compiled_graph,
        )

    def _wire_externs_or_fail(
        self,
        *,
        checked_graph: "CheckedModuleGraph",
        capabilities: "HostCapabilities",
        host_env: HostEnvironment,
        prepared: PreparedGraph,
        on_failure: "Callable[[list[Diagnostic]], _ResultT]",
    ) -> "_ResultT | None":
        """Import and resolve every extern companion, or build a failure result.

        Shared by the startup-config and run paths, which wire externs
        identically and differ only in the result dataclass returned on
        failure.  Returns ``None`` when wiring succeeds, otherwise
        ``on_failure(diagnostics)`` with the collected import/resolution
        diagnostics.
        """
        extern_diagnostics = _wire_extern_registry(
            checked_graph=checked_graph,
            capabilities=capabilities,
            registry=host_env.extern_registry,
            companion_paths=prepared.companion_paths,
        )
        if extern_diagnostics:
            return on_failure(extern_diagnostics)
        return None

    def run_prepared_graph(
        self,
        prepared: PreparedGraph,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled_graph: "MatchCompiledModuleGraph | None" = None,
        checked_graph: "CheckedModuleGraph | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
    ) -> RunResult:
        """Execute an already loaded + scoped module graph (no re-loading).

        Graph-mode analogue of :meth:`run_prepared`. Resumes the pipeline at
        type checking: ``check_graph`` → match compilation → ``lower_graph``
        → ``IrInterpreter``. Agents are entry-program-owned.

        When *prepared* carries a load/scope failure (``resolved_graph is
        None``), its diagnostics are surfaced unchanged and nothing executes.

        ``compiled_graph``
            When the caller has already typechecked and match-compiled the graph
            (for example via :meth:`discover_params_graph`), pass the result
            here to skip those static passes. ``None`` runs them here. Both
            paths still lower before the check-only stop or evaluation.
        """
        if param_values is None:
            param_values = {}

        warnings: list[Diagnostic] = list(prepared.warnings)

        if prepared.resolved_graph is None:
            return RunResult(
                ok=False,
                diagnostics=list(prepared.diagnostics),
                error=None,
                warnings=warnings,
            )
        resolved_graph = prepared.resolved_graph

        host_env = self.host_environment()
        capabilities = host_env.capabilities
        if compiled_graph is not None:
            cache_diagnostic = _cached_graph_artifact_provenance_diagnostic(
                resolved_graph, compiled_graph
            )
            if cache_diagnostic is not None:
                return RunResult(
                    ok=False,
                    diagnostics=[cache_diagnostic],
                    error=None,
                    warnings=warnings,
                )
            if compiled_graph.capabilities != capabilities:
                compiled_graph = None
        if checked_graph is not None and compiled_graph is None:
            cache_diagnostic = _cached_checked_graph_artifact_provenance_diagnostic(
                resolved_graph, checked_graph
            )
            if cache_diagnostic is not None:
                return RunResult(
                    ok=False,
                    diagnostics=[cache_diagnostic],
                    error=None,
                    warnings=warnings,
                )
            if checked_graph.capabilities != capabilities:
                checked_graph = None

        registry = host_env.registry

        # Agent reconciliation against entry module's declared agents.
        reconciliation_errors = _reconcile_agents(registry, resolved_graph.entry_agents)
        if reconciliation_errors:
            return RunResult(
                ok=False,
                diagnostics=reconciliation_errors,
                error=None,
                warnings=list(warnings),
            )

        # Reuse a supplied match-compiled graph rather than repeating its
        # typecheck and match-compilation passes.
        tc_diagnostics: tuple[Diagnostic, ...]
        if compiled_graph is not None:
            tc_diagnostics = ()
            checked_graph = compiled_graph.checked_graph
        elif checked_graph is None:
            checked_graph, tc_diagnostics = _run_typecheck_graph(resolved_graph, capabilities)
        else:
            tc_diagnostics = ()
        if checked_graph is None:
            return RunResult(
                ok=False,
                diagnostics=list(tc_diagnostics),
                error=None,
                warnings=warnings,
            )

        _append_checker_warnings(warnings, checked_graph)

        if compiled_graph is None:
            compiled_graph, match_diagnostics = _run_matchcompile_graph(checked_graph)
            if compiled_graph is None:
                return RunResult(
                    ok=False,
                    diagnostics=list(match_diagnostics),
                    error=None,
                    warnings=warnings,
                )

        contract_payloads, contract_errors = _materialize_graph_custom_contract_payloads(
            checked_graph,
            host_env.codecs,
        )
        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        if not check_only:
            # Extern (Python FFI) companions: import and resolve every declared
            # extern up front, gated by capability — fail-fast, before evaluation,
            # and after every static pass (so a static error elsewhere is reported
            # instead, with no companion import side effect). Dry-run stops before
            # this host-side import step to preserve its no-side-effects contract.
            run_failure = self._wire_externs_or_fail(
                checked_graph=checked_graph,
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
                return run_failure

        from agm.agl.lower import lower_graph

        executable = lower_graph(
            compiled_graph,
            validate=True,
            contract_payloads=contract_payloads,
        )

        return self._execute_ir(
            executable,
            host_env=host_env,
            param_values=param_values,
            check_only=check_only,
            log_file=log_file,
            warnings=warnings,
            host_settings_policy=host_settings_policy,
            builtin_host_settings=builtin_host_settings,
        )

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json

    @property
    def default_loop_limit(self) -> int | None:
        """Default max-iters valve (``None`` = OFF) for unguarded loops."""
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

        Called by ``ReplSession`` after a successful entry that contains a
        ``config`` binding, to persist the effect-at-binding for subsequent
        entries.  Agent/codec registrations and the call-depth limit are
        preserved — only the three eval-consumed settings are updated.
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
    checked: "CheckedProgramType | CheckedModuleGraph",
) -> None:
    """Append one checked artifact's warnings at the typecheck phase boundary."""
    warnings.extend(checked.warnings)


def _cached_artifact_provenance_diagnostic(
    *,
    prepared_entry: "ModuleId",
    prepared_modules: "Mapping[ModuleId, ResolvedProgram]",
    compiled_entry: "ModuleId",
    compiled_modules: "Mapping[ModuleId, ResolvedProgram]",
    span: "SourceSpan | None",
) -> Diagnostic | None:
    """Reject a cached artifact unless it wraps the exact prepared resolutions.

    Object identity is the provenance contract: structurally equal programs
    prepared in separate passes are not interchangeable compiler inputs.
    """
    same_provenance = (
        prepared_entry == compiled_entry
        and prepared_modules.keys() == compiled_modules.keys()
        and all(
            compiled_modules[module_id] is prepared_resolved
            for module_id, prepared_resolved in prepared_modules.items()
        )
    )
    if same_provenance:
        return None
    message = "Cached match-compilation artifact does not belong to the prepared source."
    if span is None:
        return Diagnostic(message=message, line=1)
    return diagnostic_from_span(message, span)


def _cached_graph_artifact_provenance_diagnostic(
    resolved_graph: "ResolvedModuleGraph",
    compiled_graph: "MatchCompiledModuleGraph",
) -> Diagnostic | None:
    """Adapt graph artifacts to the shared cached-provenance validator."""
    entry_module = resolved_graph.modules.get(resolved_graph.entry_id)
    return _cached_artifact_provenance_diagnostic(
        prepared_entry=resolved_graph.entry_id,
        prepared_modules={
            module_id: module.resolved for module_id, module in resolved_graph.modules.items()
        },
        compiled_entry=compiled_graph.checked_graph.entry_id,
        compiled_modules={
            module_id: module.resolved
            for module_id, module in compiled_graph.checked_graph.modules.items()
        },
        span=entry_module.resolved.program.span if entry_module is not None else None,
    )


def _cached_checked_artifact_provenance_diagnostic(
    *,
    prepared_entry: "ModuleId",
    prepared_modules: "Mapping[ModuleId, ResolvedProgram]",
    checked_entry: "ModuleId",
    checked_modules: "Mapping[ModuleId, ResolvedProgram]",
    span: "SourceSpan | None",
) -> Diagnostic | None:
    """Reject a checked artifact from a different prepared source."""
    return _cached_artifact_provenance_diagnostic(
        prepared_entry=prepared_entry,
        prepared_modules=prepared_modules,
        compiled_entry=checked_entry,
        compiled_modules=checked_modules,
        span=span,
    )


def _cached_checked_graph_artifact_provenance_diagnostic(
    resolved_graph: "ResolvedModuleGraph",
    checked_graph: "CheckedModuleGraph",
) -> Diagnostic | None:
    """Adapt checked graph artifacts to the shared provenance validator."""
    entry_module = resolved_graph.modules.get(resolved_graph.entry_id)
    return _cached_checked_artifact_provenance_diagnostic(
        prepared_entry=resolved_graph.entry_id,
        prepared_modules={
            module_id: module.resolved for module_id, module in resolved_graph.modules.items()
        },
        checked_entry=checked_graph.entry_id,
        checked_modules={
            module_id: module.resolved for module_id, module in checked_graph.modules.items()
        },
        span=entry_module.resolved.program.span if entry_module is not None else None,
    )


def _run_typecheck_graph(
    resolved_graph: "ResolvedModuleGraph",
    capabilities: "HostCapabilities",
) -> "tuple[CheckedModuleGraph | None, tuple[Diagnostic, ...]]":
    """Run the graph typecheck pass without raising."""
    from agm.agl.typecheck.graph import check_graph

    try:
        return check_graph(resolved_graph, capabilities), ()
    except AglError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        return None, (Diagnostic(message=f"Type error: {exc}", line=1),)


def _run_matchcompile_graph(
    checked_graph: "CheckedModuleGraph",
) -> "tuple[MatchCompiledModuleGraph | None, tuple[Diagnostic, ...]]":
    """Run whole-graph match compilation without raising."""
    from agm.agl.matchcompile import (
        MatchCompiledModuleGraph,
        compile_graph_matches,
        diagnostics_from_match_issues,
    )

    try:
        result = compile_graph_matches(checked_graph)
        if result.compiled is None:
            return None, diagnostics_from_match_issues(result.issues)
        if not isinstance(result.compiled, MatchCompiledModuleGraph):
            raise TypeError("graph match compilation returned a single-program artifact")
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


def _materialize_graph_custom_contract_payloads(
    checked_graph: "CheckedModuleGraph",
    codecs: "Mapping[str, OutputCodec]",
) -> tuple[dict[int, "ContractPayload"], list[Diagnostic]]:
    """Materialize custom contract payloads for every module in a checked graph."""
    payloads: dict[int, "ContractPayload"] = {}
    errors: list[Diagnostic] = []
    for checked_module in checked_graph.modules.values():
        module_payloads, module_errors = _materialize_custom_contract_payloads(
            checked_module.contract_specs,
            codecs,
            checked_module.type_env.type_table,
        )
        payloads.update(module_payloads)
        errors.extend(module_errors)
    return payloads, errors


def _run_typecheck(
    resolved: "ResolvedProgram",
    capabilities: "HostCapabilities",
) -> "tuple[CheckedProgramType | None, tuple[Diagnostic, ...]]":
    """Run the typecheck pass without raising."""
    from agm.agl.typecheck import check

    try:
        return check(resolved, capabilities), ()
    except AglError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        return None, (Diagnostic(message=f"Type error: {exc}", line=1),)


def _run_matchcompile(
    checked: "CheckedProgramType",
) -> "tuple[MatchCompiledProgram | None, tuple[Diagnostic, ...]]":
    """Run single-program match compilation without raising."""
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
            raise TypeError("single match compilation returned a graph artifact")
        return result.compiled, ()
    except Exception as exc:
        return None, (Diagnostic(message=f"Match compilation error: {exc}", line=1),)


def _extern_declaration_sort_key(pair: "tuple[ModuleId, str]") -> "tuple[tuple[str, ...], str]":
    """Sort key for deterministic ``(module_id, name)`` diagnostic ordering."""
    return (pair[0].segments, pair[1])


def _extern_declarations(
    checked_graph: "CheckedModuleGraph",
) -> list[tuple["ModuleId", str]]:
    """Return ``(module_id, name)`` for every ``extern def`` in *checked_graph*.

    Sorted for deterministic diagnostic ordering; the checked AST (not the
    IR) is the source of truth here since the IR has no extern table yet.
    """
    declarations: list[tuple[ModuleId, str]] = [
        (mid, name)
        for mid, mod in checked_graph.modules.items()
        for name, funcdef in mod.resolved.declared_functions.items()
        if funcdef.is_extern
    ]
    declarations.sort(key=_extern_declaration_sort_key)
    return declarations


def _wire_extern_registry(
    *,
    checked_graph: "CheckedModuleGraph",
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
    declarations = _extern_declarations(checked_graph)

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
