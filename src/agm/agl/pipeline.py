"""PipelineDriver — top-of-stack orchestrator for the AgL execution pipeline.

Drives the full ``parse → scope → typecheck → lower/link → IR eval`` pipeline:
registers agents/codecs, validates host params, materializes output
contracts, and executes the program (or stops after static checking for
``agm exec --dry-run``).  Structured outputs use the JSON codec with
lenient-by-default recovery (design §2.8).

``agm.agl.runtime`` is the eval-free services layer (agents, codecs, params,
types).  This module is the top-of-stack host façade that depends on both
``runtime`` services and ``agm.agl.eval``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
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
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.scope.graph import ResolvedModuleGraph
    from agm.agl.scope.symbols import ResolvedProgram
    from agm.agl.semantics.values import ExceptionValue, Value
    from agm.agl.syntax.nodes import AgentDecl as AgentDeclNode
    from agm.agl.syntax.nodes import PragmaValue, Program
    from agm.agl.typecheck.env import CheckedProgram as CheckedProgramType
    from agm.agl.typecheck.graph import CheckedModuleGraph

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
    def config_pragmas(self) -> "dict[str, PragmaValue]":
        """The validated ``config`` pragmas declared in source.

        Returns a mapping of key → value for each pragma collected by the scope
        pass.  Empty when parse or scope failed (``resolved is None``), or when
        the program declares no ``config`` pragmas.
        """
        if self.resolved is None:
            return {}
        return dict(self.resolved.config_pragmas)

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
    """

    source: str
    entry_path: "Path | None"
    roots: "RootSet"
    resolved_graph: "ResolvedModuleGraph | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]

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
    def config_pragmas(self) -> "dict[str, PragmaValue]":
        """The validated ``config`` pragmas declared in the entry module.

        Empty when load or scope failed (``resolved_graph is None``).
        """
        from agm.agl.modules.ids import ENTRY_ID

        if self.resolved_graph is None:
            return {}
        entry_mod = self.resolved_graph.modules.get(ENTRY_ID)
        if entry_mod is None:
            return {}
        return dict(entry_mod.resolved.config_pragmas)

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
    §12.6: source location is part of runtime error reporting); ``None`` when
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
        §12.6) and the REPL failure echo so the two never diverge.
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
        lex/parse/scope/typecheck/param-validation.  Each entry has a
        ``.message`` (str) and a ``.line`` (int, 1-based).  Warnings are a
        SEPARATE channel and NEVER appear here; on a successful run this list is
        empty.
    ``warnings``
        Advisory warning-severity diagnostics (e.g. non-exhaustive ``case``)
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
        This is the handle referred to in plan §8.3.
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
    default_loop_limit : int
        Default iteration bound for ``do[N]`` loops (design §2.11).
    default_strict_json : bool
        When ``True`` the JSON codec defaults to strict parsing (only a bare
        JSON value with surrounding whitespace is accepted).  The default
        ``False`` enables lenient JSON recovery (design §2.8, Q3).
    default_agent : callable or None
        The callable used for the built-in ``ask`` agent.  ``None`` means
        no default agent is configured (only explicitly registered agents will
        be available).
    shell_exec_timeout : float or None
        Idle timeout (in seconds) applied to every ``exec`` shell call (design
        §4.12, §11.13).  ``None`` means no timeout (the shell command may run
        indefinitely).  This is the ``[exec] timeout`` config value, threaded
        in from the CLI (plan §10.3).
    default_call_depth_limit : int
        Maximum call depth for recursive functions (design §D8).  Exceeding
        this limit raises a ``RecursionError`` in the AgL program.  Default
        is 256.
    """

    def __init__(
        self,
        *,
        default_loop_limit: int = 5,
        default_strict_json: bool = False,
        default_agent: AgentFn | None = None,
        shell_exec_timeout: float | None = None,
        default_call_depth_limit: int = 256,
    ) -> None:
        self._default_loop_limit = default_loop_limit
        self._default_strict_json = default_strict_json
        self._default_agent = default_agent
        self._shell_exec_timeout = shell_exec_timeout
        self._default_call_depth_limit = default_call_depth_limit
        self._agents: dict[str, AgentFn] = {}
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
        can validate ``format`` options at a call site (design §7.6 / CARRY-IN 1).

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
                return PreparedProgram(
                    source, None, None, (exc.to_diagnostic(),), tuple(tab_sink)
                )
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
            return PreparedProgram(
                source, program, None, (exc.to_diagnostic(),), warnings
            )
        except Exception as exc:
            return PreparedProgram(
                source,
                program,
                None,
                (Diagnostic(message=f"Scope error: {exc}", line=1),),
                warnings,
            )

        # Scope warnings (e.g. a declared-but-uncalled agent) join the lex ones.
        return PreparedProgram(
            source, program, resolved, (), (*warnings, *resolved.warnings)
        )

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
        """Typecheck-only discovery for typed ``param`` declarations."""
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
        all_warnings = (*prepared.warnings, *(checked.warnings if checked is not None else ()))
        if checked is None:
            return ParamDiscovery(
                params=(),
                program_name=prepared.program_name,
                checked=None,
                diagnostics=tc_diagnostics,
                warnings=all_warnings,
            )

        assert prepared.program is not None
        infos: list[ParamDeclInfo] = []
        for item in prepared.program.body.items:
            if isinstance(item, ParamDecl):
                param_type = checked.type_env.get_binding_type(item.node_id)
                assert param_type is not None, (
                    f"Param {item.name!r} has no recorded binding type; "
                    "checker invariant violated."
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
            parse → resolve → check (with HostCapabilities) →
            validate params → materialize contracts → eval

        Convenience wrapper: ``run(source)`` is exactly
        ``run_prepared(prepare(source))``.  A host that needs the declared-agent
        inventory before execution should call :meth:`prepare` once and pass the
        result to :meth:`run_prepared`, so the source is parsed and scoped only
        once.

        When ``check_only`` is ``True`` (``agm exec --dry-run``) the runtime
        runs the full static pipeline, param validation, and contract
        materialization, then STOPS before executing any statement: a clean
        program returns ``ok=True`` with no bindings and produces no program
        output; static/param errors still return ``ok=False``.  On a clean
        ``check_only`` run the §10.1 static call-site inventory is populated on
        ``RunResult.call_sites`` (printed by ``agm exec --dry-run``).

        ``log_file`` is the path of the JSONL trace file to write.  When
        ``None`` (the default) no trace is written.  Dry-run (``check_only``)
        never writes a trace regardless of *log_file* (plan §10.1: dry-run
        is side-effect-free).

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
        checked: "CheckedProgramType | None" = None,
    ) -> RunResult:
        """Execute an already parsed + scoped program (no re-parsing).

        Resumes the pipeline at type checking: reconcile agents → check →
        validate params → materialize contracts → eval.  See :meth:`run` for the
        ``check_only`` / ``log_file`` semantics.  When *prepared* carries a
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

        # ----------------------------------------------------------------
        # [2b] Source↔host agent reconciliation (plan §8, decisions 1 & 11)
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
        if checked is None:
            checked, tc_diagnostics = _run_typecheck(resolved, capabilities)
            if checked is None:
                return RunResult(
                    ok=False,
                    diagnostics=list(tc_diagnostics),
                    error=None,
                    warnings=warnings,
                )

        # Collect warnings from typecheck.
        warnings.extend(checked.warnings)

        from agm.agl.lower import lower_program

        executable = lower_program(
            checked,
            source_text=source,
            source_label="<entry>",
            validate=True,
        )

        ir_param_values, param_errors = _prepare_ir_params(executable, param_values)
        if param_errors:
            return RunResult(ok=False, diagnostics=param_errors, error=None, warnings=warnings)

        host_contracts, contract_errors = _materialize_ir_contracts(
            executable, host_env.codecs
        )
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
        # before executing any statement (no program output, no side effects).
        # Build the §10.1 static call-site inventory before returning.
        # Dry-run is side-effect-free: no trace is written (plan §10.1).
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
        # [7] Build and run the interpreter
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

        interp = IrInterpreter(
            executable,
            registry=registry,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            max_call_depth=self._default_call_depth_limit,
            param_values=ir_param_values,
            host_contracts=host_contracts,
        )

        try:
            root_bindings = interp.run()
        except AglRaise as exc:
            # Uncaught AgL exception (exit code 2 per the CLI contract).
            # ONLY the AgL exception carrier is caught here: an unexpected Python
            # exception is an interpreter bug and must propagate (crash loudly)
            # rather than masquerade as a user-facing pre-execution diagnostic.
            error = exception_value_to_run_error(exc.exc, span=exc.span)
            # Record the uncaught exception in the trace (design §12.6: include
            # the source span when the raise site threaded it through AglRaise).
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
            ) as exc:
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
        return PreparedGraph(
            entry_source, entry_path, roots, resolved_graph, (), all_warnings
        )

    def discover_params_graph(self, prepared: PreparedGraph) -> ParamDiscovery:
        """Typecheck-only discovery for typed ``param`` declarations in a graph.

        Graph-mode analogue of :meth:`discover_params`.  Reads typed param
        declarations from the entry module of *prepared.resolved_graph*.
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
        checked_graph, tc_diagnostics = _run_typecheck_graph(
            prepared.resolved_graph, capabilities
        )
        all_warnings_list: list[Diagnostic] = list(prepared.warnings)
        if checked_graph is not None:
            all_warnings_list.extend(checked_graph.warnings)
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
        )

        infos: list[ParamDeclInfo] = []
        for item in entry_cm.resolved.program.body.items:
            if isinstance(item, ParamDecl):
                param_type = entry_cm.type_env.get_binding_type(item.node_id)
                assert param_type is not None, (
                    f"Param {item.name!r} has no recorded binding type; "
                    "checker invariant violated."
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
        )

    def run_prepared_graph(
        self,
        prepared: PreparedGraph,
        *,
        param_values: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
        checked_graph: "CheckedModuleGraph | None" = None,
    ) -> RunResult:
        """Execute an already loaded + scoped module graph (no re-loading).

        Graph-mode analogue of :meth:`run_prepared`.  Resumes the pipeline at
        type checking: ``check_graph`` → ``lower_graph`` → ``IrInterpreter``.
        Agents are entry-program-owned.

        When *prepared* carries a load/scope failure (``resolved_graph is
        None``), its diagnostics are surfaced unchanged and nothing executes.

        ``checked_graph``
            When the caller has already type-checked the graph (e.g. via
            :meth:`discover_params_graph`), pass the result here to skip the
            second redundant ``check_graph`` call.  ``None`` (the default)
            runs type checking here as before.
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
        registry = host_env.registry
        capabilities = host_env.capabilities

        # Agent reconciliation against entry module's declared agents.
        reconciliation_errors = _reconcile_agents(registry, resolved_graph.entry_agents)
        if reconciliation_errors:
            return RunResult(
                ok=False,
                diagnostics=reconciliation_errors,
                error=None,
                warnings=list(warnings),
            )

        # Type checking — skip if the caller already has a checked graph
        # (e.g. from discover_params_graph) to avoid running check_graph twice.
        if checked_graph is None:
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

        warnings.extend(checked_graph.warnings)

        from agm.agl.lower import lower_graph

        executable = lower_graph(checked_graph, validate=True)

        ir_param_values, param_errors = _prepare_ir_params(executable, param_values)
        if param_errors:
            return RunResult(ok=False, diagnostics=param_errors, error=None, warnings=warnings)

        host_contracts, contract_errors = _materialize_ir_contracts(
            executable, host_env.codecs
        )
        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        # Dry-run stop: build call-site inventory from entry module only.
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

        # Execute the graph.
        from agm.agl.runtime.trace import TraceStore
        from agm.agl.semantics.exceptions import AglRaise

        trace = TraceStore(path=log_file)
        if log_file is not None:
            from agm.core.fs import mkdir

            mkdir(log_file.parent, parents=True, exist_ok=True)
        trace.run_start()

        try:
            root_bindings = IrInterpreter(
                executable,
                registry=registry,
                loop_limit=self._default_loop_limit,
                strict_json=self._default_strict_json,
                shell_exec_timeout=self._shell_exec_timeout,
                trace=trace,
                max_call_depth=self._default_call_depth_limit,
                param_values=ir_param_values,
                host_contracts=host_contracts,
            ).run()
        except AglRaise as exc:
            error = exception_value_to_run_error(exc.exc, span=exc.span)
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

    @property
    def default_loop_limit(self) -> int:
        """Default iteration bound for ``do[N]`` loops."""
        return self._default_loop_limit

    @property
    def default_strict_json(self) -> bool:
        """Whether strict JSON parsing is the default."""
        return self._default_strict_json

    @property
    def shell_exec_timeout(self) -> float | None:
        """Idle timeout in seconds for ``exec`` shell calls (``None`` = no timeout)."""
        return self._shell_exec_timeout

    @property
    def default_call_depth_limit(self) -> int:
        """Maximum call depth for recursive functions."""
        return self._default_call_depth_limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_typecheck_graph(
    resolved_graph: "ResolvedModuleGraph",
    capabilities: "HostCapabilities",
) -> "tuple[CheckedModuleGraph | None, tuple[Diagnostic, ...]]":
    """Run the graph typecheck pass without raising."""
    from agm.agl.typecheck.env import AglTypeError
    from agm.agl.typecheck.graph import check_graph

    try:
        return check_graph(resolved_graph, capabilities), ()
    except AglTypeError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        return None, (Diagnostic(message=f"Type error: {exc}", line=1),)


def _run_typecheck(
    resolved: "ResolvedProgram",
    capabilities: "HostCapabilities",
) -> "tuple[CheckedProgramType | None, tuple[Diagnostic, ...]]":
    """Run the typecheck pass without raising."""
    from agm.agl.typecheck import AglTypeError, check

    try:
        return check(resolved, capabilities), ()
    except AglTypeError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        return None, (Diagnostic(message=f"Type error: {exc}", line=1),)


def _reconcile_agents(
    registry: "AgentRegistry",
    declared_agents: "Mapping[str, AgentDeclNode]",
) -> list[Diagnostic]:
    """Enforce the source↔host agent contract (plan §8, decisions 1 & 11).

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
        codec_kinds={name: codec.supported_kinds for name, codec in all_codecs.items()},
    )
    return HostEnvironment(
        registry=registry,
        capabilities=capabilities,
        codecs=all_codecs,
    )


def exception_value_to_run_error(
    exc: "ExceptionValue",
    *,
    span: "object" = None,  # SourceSpan | None — avoids import cycle
) -> RunError:
    """Convert an ``ExceptionValue`` to a ``RunError`` for ``RunResult``.

    Field values are converted via the shared serializer, which preserves
    ``Decimal`` exactness (never routed through binary ``float``; design §5.1).

    *span* is the optional raise-site source span threaded from ``AglRaise``
    (design §12.6); when present, ``RunError.line`` and ``RunError.col`` are
    populated from it so the CLI can include the source location in its
    exit-2 error output.
    """
    from agm.agl.ir.ids import Location
    from agm.agl.runtime.serialize import value_to_json_obj
    from agm.agl.syntax.spans import SourceSpan

    fields: dict[str, object] = {
        k: value_to_json_obj(v) for k, v in exc.fields.items()
    }
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
