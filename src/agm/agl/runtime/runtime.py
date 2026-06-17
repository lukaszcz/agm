"""WorkflowRuntime — the public façade for the AgL host runtime.

Drives the full ``parse → scope → typecheck → host-prep → eval`` pipeline:
registers agents/codecs, validates host inputs, materializes output
contracts, and executes the program (or stops after static checking for
``agm exec --dry-run``).  Structured outputs use the JSON codec with
lenient-by-default recovery (design §2.8).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agl.diagnostics import Diagnostic
from agm.agl.runtime.agents import AgentFn

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.capabilities import HostCapabilities
    from agm.agl.eval.values import ExceptionValue, Value
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.scope.symbols import ResolvedProgram
    from agm.agl.syntax.nodes import AgentDecl as AgentDeclNode
    from agm.agl.syntax.nodes import PragmaValue, Program
    from agm.agl.typecheck.env import CheckedProgram as CheckedProgramType
    from agm.agl.typecheck.types import Type as AglType

# Reserved agent names: cannot be registered by callers.
_RESERVED_AGENT_NAMES: frozenset[str] = frozenset({"ask", "exec"})


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """Assembled host-runtime environment shared by ``run`` and the REPL session.

    Bundles the three pieces that both the whole-program runner
    (``WorkflowRuntime.run``) and the incremental ``ReplSession`` need to build
    identically from a set of agent/codec registrations:

    ``registry``
        The ``AgentRegistry`` (named agents + optional default agent).
    ``capabilities``
        The ``HostCapabilities`` static catalog derived from the registry and
        codecs — consumed by the type checker.
    ``codecs``
        The merged ``name → OutputCodec`` table (built-ins + host extras),
        used for contract materialization.
    """

    registry: "AgentRegistry"
    capabilities: "HostCapabilities"
    codecs: dict[str, "OutputCodec"]


@dataclass(frozen=True, slots=True)
class CallSiteInfo:
    """Static summary of one agent-call or exec site (--dry-run inventory).

    ``callee``        Agent or executor name (``"ask"``, ``"exec"``, or a
                      registered agent name).
    ``target_type``   The target type name (e.g. ``"text"``, ``"Review"``).
    ``codec_name``    Selected codec (``"text"`` or ``"json"``).
    ``has_schema``    ``True`` when the contract carries a JSON Schema.
    ``parse_policy``  ``"abort"`` / ``"retry[N]"`` / ``"default"``.
    ``line``          1-based source line of the call site.
    ``col``           1-based source column of the call site.
    """

    callee: str
    target_type: str
    codec_name: str
    has_schema: bool
    parse_policy: str
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class AgentDeclInfo:
    """Static summary of one ``agent`` declaration in a program.

    ``name``
        The declared agent name.
    ``runner``
        The optional static runner-command hint (a literal string with NO
        interpolation), or ``None`` for a bare ``agent NAME`` declaration.
    ``line``
        1-based source line of the declaration (``span.start_line``).
    ``col``
        1-based source column of the declaration (``span.start_col``).
    """

    name: str
    runner: str | None
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class PreparedProgram:
    """Result of the lex + parse + scope phase of an AgL program.

    Produced by :meth:`WorkflowRuntime.prepare` and consumed by
    :meth:`WorkflowRuntime.run_prepared`, so those two static phases run exactly
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
    """Result of a ``WorkflowRuntime.run`` call.

    ``ok``
        ``True`` iff there are no error-severity ``diagnostics`` **and** no
        uncaught AgL exception.  ``warnings`` never affect ``ok``.
    ``diagnostics``
        Pre-execution FAILURES only: error-severity items from
        lex/parse/scope/typecheck/input-validation.  Each entry has a
        ``.message`` (str) and a ``.line`` (int, 1-based).  Warnings are a
        SEPARATE channel and NEVER appear here; on a successful run this list is
        empty.
    ``warnings``
        Advisory warning-severity diagnostics (e.g. non-exhaustive ``case``)
        surfaced on EVERY path — success, static failure, input-validation
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


class WorkflowRuntime:
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
        return WorkflowRuntime.prepare(source).declared_agents

    def run(
        self,
        source: str,
        *,
        inputs: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
    ) -> RunResult:
        """Parse, analyse, and (unless ``check_only``) execute an AgL program.

        Pipeline:
            parse → resolve → check (with HostCapabilities) →
            validate inputs → materialize contracts → eval

        Convenience wrapper: ``run(source)`` is exactly
        ``run_prepared(prepare(source))``.  A host that needs the declared-agent
        inventory before execution should call :meth:`prepare` once and pass the
        result to :meth:`run_prepared`, so the source is parsed and scoped only
        once.

        When ``check_only`` is ``True`` (``agm exec --dry-run``) the runtime
        runs the full static pipeline, input validation, and contract
        materialization, then STOPS before executing any statement: a clean
        program returns ``ok=True`` with no bindings and produces no program
        output; static/input errors still return ``ok=False``.  On a clean
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
            inputs=inputs,
            check_only=check_only,
            log_file=log_file,
        )

    def run_prepared(
        self,
        prepared: PreparedProgram,
        *,
        inputs: Mapping[str, object] | None = None,
        check_only: bool = False,
        log_file: "Path | None" = None,
    ) -> RunResult:
        """Execute an already parsed + scoped program (no re-parsing).

        Resumes the pipeline at type checking: reconcile agents → check →
        validate inputs → materialize contracts → eval.  See :meth:`run` for the
        ``check_only`` / ``log_file`` semantics.  When *prepared* carries a
        captured parse/scope failure (``resolved is None``), its diagnostics are
        surfaced unchanged and nothing executes.
        """
        if inputs is None:
            inputs = {}

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
        # Reported on the same channel as input-validation / host-config
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
        from agm.agl.typecheck import AglTypeError, check

        try:
            checked = check(resolved, capabilities)
        except AglTypeError as exc:
            return RunResult(
                ok=False,
                diagnostics=[exc.to_diagnostic()],
                error=None,
                warnings=warnings,
            )
        except Exception as exc:
            return RunResult(
                ok=False,
                diagnostics=[Diagnostic(message=f"Type error: {exc}", line=1)],
                error=None,
                warnings=warnings,
            )

        # Collect warnings from typecheck.
        warnings.extend(checked.warnings)

        # ----------------------------------------------------------------
        # [4] Validate host inputs against input declarations
        # ----------------------------------------------------------------
        from agm.agl.syntax.nodes import InputDecl

        # Build declared input map.  Read the exact binding type recorded by the
        # checker (keyed by the InputDecl node_id) rather than re-resolving the
        # annotation here — the checker is the single source of truth and already
        # handles compound types (list/dict/record/enum) correctly.
        declared_inputs: dict[str, AglType] = {}  # name → declared Type
        # Track each declaration's source line so the "missing declared input"
        # diagnostic can report the declaration site (parity with the
        # type-invalid path, which already uses ``stmt.span``).
        declared_input_lines: dict[str, int] = {}  # name → declaration line
        for stmt in program.body.items:
            if isinstance(stmt, InputDecl):
                input_type = checked.type_env.get_binding_type(stmt.node_id)
                if input_type is None:
                    raise AssertionError(
                        f"Input {stmt.name!r} has no recorded binding type; "
                        "checker invariant violated."
                    )
                declared_inputs[stmt.name] = input_type
                declared_input_lines[stmt.name] = stmt.span.start_line

        # Validate: check for missing and undeclared.
        input_errors: list[Diagnostic] = []
        provided_keys = set(inputs.keys())
        declared_keys = set(declared_inputs.keys())

        for name in declared_keys - provided_keys:
            input_errors.append(
                Diagnostic(
                    message=f"Missing declared input: {name!r}",
                    line=declared_input_lines[name],
                )
            )
        for name in provided_keys - declared_keys:
            input_errors.append(
                Diagnostic(
                    message=f"Undeclared input: {name!r} was provided but not declared",
                    line=1,
                )
            )

        if input_errors:
            return RunResult(
                ok=False,
                diagnostics=input_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [5] Materialize output contracts (text codec only in M1)
        # ----------------------------------------------------------------
        from agm.agl.runtime.contract import materialize_contract

        # Reuse the merged codec map built for HostCapabilities above.
        codecs = host_env.codecs
        contracts: dict[int, object] = {}
        contract_errors: list[Diagnostic] = []

        for node_id, spec in checked.contract_specs.items():
            try:
                contracts[node_id] = materialize_contract(spec, codecs)
            except ValueError as exc:
                contract_errors.append(
                    Diagnostic(message=f"Contract error: {exc}", line=1)
                )

        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [6] Build root scope from inputs + type-check declarations
        # ----------------------------------------------------------------
        from agm.agl.eval.scope import Scope

        root_scope = Scope(parent=None)
        input_bind_errors: list[Diagnostic] = []

        for stmt in program.body.items:
            if isinstance(stmt, InputDecl):
                raw_val = inputs[stmt.name]
                input_type_obj = declared_inputs[stmt.name]
                # Convert/validate the raw value.
                try:
                    typed_val = convert_input(stmt.name, raw_val, input_type_obj)
                except ValueError as exc:
                    input_bind_errors.append(
                        Diagnostic(message=str(exc), line=stmt.span.start_line)
                    )
                    continue
                root_scope.define(
                    stmt.name, typed_val, mutable=False, decl_span=stmt.span
                )

        if input_bind_errors:
            return RunResult(
                ok=False,
                diagnostics=input_bind_errors,
                error=None,
                warnings=list(warnings),
            )

        # ----------------------------------------------------------------
        # [check_only] --dry-run stop: the full static pipeline, input
        # validation, and contract materialization have all succeeded.  Stop
        # before executing any statement (no program output, no side effects).
        # Build the §10.1 static call-site inventory before returning.
        # Dry-run is side-effect-free: no trace is written (plan §10.1).
        # ----------------------------------------------------------------
        if check_only:
            inventory = _build_call_inventory(checked, contracts)
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
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.interpreter import Interpreter
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.runtime.trace import TraceStore

        # Create the trace store for this run.  When log_file is None the
        # store is a no-op and no file is touched.
        trace = TraceStore(path=log_file)
        if log_file is not None:
            from agm.core.fs import mkdir

            mkdir(log_file.parent, parents=True, exist_ok=True)
        trace.run_start()

        typed_contracts: dict[int, OutputContract] = {
            nid: c for nid, c in contracts.items() if isinstance(c, OutputContract)
        }

        interp = Interpreter(
            checked=checked,
            registry=registry,
            contracts=typed_contracts,
            type_env=checked.type_env,
            loop_limit=self._default_loop_limit,
            strict_json=self._default_strict_json,
            source=source,
            shell_exec_timeout=self._shell_exec_timeout,
            trace=trace,
            max_call_depth=self._default_call_depth_limit,
        )

        try:
            interp.execute(root_scope)
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

        # Successful run: snapshot root bindings.
        root_bindings = root_scope.snapshot()
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
    ``HostCapabilities`` exactly as ``WorkflowRuntime.run`` did inline.
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


def convert_input(name: str, raw: object, type_obj: "AglType") -> "Value":
    """Convert a raw host input value to the declared AgL type.

    Supported types:
    - ``text``: verbatim (the value must already be a ``str``).
    - ``int``/``decimal``/``bool``/``json``: parsed via stdlib ``json`` with
      ``parse_float=Decimal`` (design §5.1: no binary floats) and validated.
    - ``list``/``dict``/``record``/``enum``: parsed from a JSON string via the
      ``JsonCodec`` (M2+).
    """
    import decimal as _decimal

    from agm.agl.eval.values import (
        BoolValue,
        DecimalValue,
        IntValue,
        JsonValue,
        TextValue,
    )
    from agm.agl.typecheck.types import (
        BoolType,
        DecimalType,
        DictType,
        EnumType,
        IntType,
        JsonType,
        ListType,
        RecordType,
        TextType,
    )

    # Text: verbatim.
    if isinstance(type_obj, TextType):
        if not isinstance(raw, str):
            raise ValueError(
                f"Input {name!r}: expected a text value (str), got {type(raw).__name__}"
            )
        return TextValue(raw)

    # Structured types (list/dict/record/enum): delegate to JsonCodec.
    if isinstance(type_obj, (ListType, DictType, RecordType, EnumType)):
        from agm.agl.runtime.codec import JsonCodec
        from agm.agl.runtime.schema import derive_schema

        # Accept either a JSON string or a Python native object (dict/list/
        # scalar) that was already parsed from JSON.  For native objects we
        # re-serialize to a JSON string so the codec can validate and convert
        # using the full type-aware path.  Decimal values are serialized
        # losslessly using dumps_exact, which emits them as unquoted numeric
        # text so the codec's json.loads(parse_float=Decimal) round-trip is
        # exact — avoiding the old default=str bug that turned Decimal("1.5")
        # into the JSON string "1.5" and failed schema validation.
        if isinstance(raw, str):
            json_str = raw
        elif _is_json_shaped(raw):
            from agm.agl.runtime.serialize import dumps_exact

            json_str = dumps_exact(raw, indent=None)
        else:
            raise ValueError(
                f"Input {name!r} has type {type_obj!r}; structured inputs must be "
                "provided as a JSON string or a JSON-compatible Python value "
                f"(got {type(raw).__name__!r})."
            )
        codec = JsonCodec()
        # Precompute schema once (CARRY-IN 2: avoids re-derivation inside parse).
        schema = derive_schema(type_obj)
        # Host-supplied --input values are not chatty agent output: they must be
        # exactly one bare JSON value (F7).  Strict parsing avoids json-repair
        # silently "fixing" user typos.
        result = codec.parse(json_str, type_obj, strict_json=True, schema=schema)
        if not result.ok or result.value is None:
            raise ValueError(
                f"Input {name!r}: could not parse as {type_obj!r}; structured "
                f"inputs must be exactly one valid JSON value: {result.error_msg}"
            )
        return result.value

    # Scalar non-text (int/decimal/bool/json): parse from JSON if given as string.
    if not isinstance(type_obj, (IntType, DecimalType, BoolType, JsonType)):
        raise ValueError(
            f"Input {name!r} has unsupported type {type_obj!r}."
        )

    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_float=_decimal.Decimal)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Input {name!r}: could not parse as JSON: {exc}"
            ) from exc

    if isinstance(type_obj, IntType):
        if isinstance(value, int) and not isinstance(value, bool):
            return IntValue(value)
        if isinstance(value, _decimal.Decimal) and value == int(value):
            return IntValue(int(value))
        raise ValueError(
            f"Input {name!r}: expected an integer, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, DecimalType):
        if isinstance(value, _decimal.Decimal):
            return DecimalValue(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return DecimalValue(_decimal.Decimal(value))
        raise ValueError(
            f"Input {name!r}: expected a decimal, got {type(value).__name__} {value!r}"
        )

    if isinstance(type_obj, BoolType):
        if isinstance(value, bool):
            return BoolValue(value)
        raise ValueError(
            f"Input {name!r}: expected a bool, got {type(value).__name__} {value!r}"
        )

    # JsonType: accept any parsed JSON value.
    return JsonValue(value)


def _is_json_shaped(obj: object) -> bool:
    """Return ``True`` iff *obj* is a JSON-compatible Python value.

    The closed set: ``None``, ``bool``, ``int``, ``float``,
    ``decimal.Decimal``, ``str``, ``list`` (elements recursively JSON-shaped),
    and ``dict`` (str keys, values recursively JSON-shaped).

    Used by :func:`convert_input` to detect non-JSON-shaped host objects
    (e.g. sets or custom classes) before attempting serialisation, so the
    caller can emit a clean diagnostic instead of a cryptic traceback.
    """
    import decimal as _decimal_mod

    if obj is None or isinstance(obj, (bool, int, float, str, _decimal_mod.Decimal)):
        return True
    if isinstance(obj, list):
        return all(_is_json_shaped(e) for e in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_shaped(v) for k, v in obj.items())
    return False


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
    from agm.agl.runtime.serialize import value_to_json_obj
    from agm.agl.syntax.spans import SourceSpan

    fields: dict[str, object] = {
        k: value_to_json_obj(v) for k, v in exc.fields.items()
    }
    line: int | None = None
    col: int | None = None
    if isinstance(span, SourceSpan):
        line = span.start_line
        col = span.start_col
    return RunError(type_name=exc.type_name, fields=fields, line=line, col=col)


def _build_call_inventory(
    checked: CheckedProgramType,
    contracts: dict[int, object],
) -> list[CallSiteInfo]:
    """Build the §10.1 static call-site inventory from the checked program.

    The inventory is derived entirely from the checker's work: each
    ``CallSiteRecord`` (recorded in source order while type-checking) supplies the
    callee, parse policy, and span; ``contract_specs`` supplies the codec and
    target type; and the materialized ``contracts`` table supplies schema
    presence.  No second AST walk is performed.

    Returns one ``CallSiteInfo`` per agent-call/exec site, in source order.
    """
    from agm.agl.runtime.contract import OutputContract

    inventory: list[CallSiteInfo] = []

    for record in checked.call_sites:
        # The checker records a CallSiteRecord and an OutputContractSpec together
        # in ``_check_agent_call`` (both keyed by the call's node_id), so every
        # recorded call site has a spec.  A missing spec is a checker-invariant
        # violation, not a normal skip.
        spec = checked.contract_specs[record.node_id]

        contract = contracts.get(record.node_id)
        has_schema = (
            isinstance(contract, OutputContract) and contract.json_schema is not None
        )

        inventory.append(
            CallSiteInfo(
                callee=record.callee,
                target_type=repr(spec.target_type),
                codec_name=spec.codec_name,
                has_schema=has_schema,
                parse_policy=record.parse_policy,
                line=record.line,
                col=record.col,
            )
        )

    return inventory
