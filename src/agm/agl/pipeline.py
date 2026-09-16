"""PipelineDriver — top-of-stack orchestrator for the AgL execution pipeline.

Drives the full ``parse → scope → typecheck → matchcompile → lower/link → IR eval`` pipeline:
registers agents/codecs, validates host-supplied program arguments, materializes
output contracts, and executes the program (or stops after static checking for
``agm exec --dry-run``).  Structured outputs use the JSON codec with
lenient-by-default recovery.

``agm.agl.runtime`` is the eval-free services layer (agents, codecs, arguments,
types).  This module is the top-of-stack host façade that depends on both
``runtime`` services and ``agm.agl.eval``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeVar

from agm.agl.artifact_cache import (
    retain_match_sites,
    retained_match_sites,
    retained_module_sources,
)
from agm.agl.diagnostics import AglError, Diagnostic, diagnostic_from_span
from agm.agl.eval.ir_interpreter import (
    HostConfigurationError,
    IrInterpreter,
)
from agm.agl.ir.nodes import UseDefault
from agm.agl.recursion import NestingTooDeepError, frontend_recursion_boundary
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.contract import materialize_ir_contracts
from agm.agl.runtime.types import (
    CallSiteInfo as CallSiteInfo,
)
from agm.agl.runtime.types import (
    HostEnvironment,
    ParamBindingInfo,
    ProgramDeclInfo,
    ProgramParamInfo,
)
from agm.agl.self_validation import self_validation_enabled

if TYPE_CHECKING:
    from pathlib import Path

    from agm.agl.capabilities import HostCapabilities
    from agm.agl.ir.builtin_vars import BuiltinVarKey
    from agm.agl.ir.contracts import ContractPayload, ExceptionFieldEncode
    from agm.agl.ir.ids import FunctionId, NominalId, SymbolId
    from agm.agl.ir.program import ExecutableProgram, FunctionDescriptor, NominalDescriptor
    from agm.agl.ir.static_keys import StaticBindingKey
    from agm.agl.matchcompile import MatchCompiledProgram
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import ModuleGraph
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.arguments import ProgramArguments
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.runtime.sessions import SessionHost
    from agm.agl.scope.program import ResolvedProgram
    from agm.agl.scope.symbols import ModuleResolution
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.types import Type
    from agm.agl.semantics.values import ExceptionValue, Value
    from agm.agl.syntax.advisories import SpacedQualifier
    from agm.agl.syntax.nodes import FuncDef, Program
    from agm.agl.syntax.types import TypeExpr
    from agm.agl.typecheck.env import OutputContractSpec
    from agm.agl.typecheck.program import CheckedProgram
    from agm.packages.model import PackageInfo

_ResultT = TypeVar("_ResultT")


class ArtifactProvenanceError(Exception):
    """A cached compiler artifact does not belong to the prepared source it is
    handed back with.

    The artifact seam is internal — a caller passes back an artifact this
    pipeline produced for a specific prepared source — so a mismatch is a
    host-wiring bug with no user-facing remedy, not a diagnostic about the
    program. Frontend artifacts are re-verified by optional self-validation;
    lowered executables are validated by the pipeline that issued them.
    """


@dataclass(frozen=True, slots=True)
class _ExecutableProvenance:
    """Pipeline-owned cache metadata kept out of the typeless execution IR.

    ``executable`` is never read: it keeps the executable alive so its
    ``id()``, which keys this cache, cannot be recycled by a later object.
    """

    executable: "ExecutableProgram"
    prepared: "PreparedProgram"
    capabilities: "HostCapabilities"
    selected_module: "ModuleId"


@dataclass(frozen=True, slots=True)
class ProgramDiscovery:
    """Result of ``PipelineDriver.discover_programs``.

    Reports every ``program def`` declaration with its own typed
    value-parameter signature (``ProgramDeclInfo.parameters``), plus each
    checked module's static parameter inventory. :meth:`params_for` projects
    that inventory through one program's closure without another static pass.
    """

    programs: tuple[ProgramDeclInfo, ...]
    checked: "CheckedProgram | None"
    compiled: "MatchCompiledProgram | None"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]
    module_params: Mapping["ModuleId", tuple[ParamBindingInfo, ...]] = field(default_factory=dict)

    def params_for(self, program: ProgramDeclInfo) -> tuple[ParamBindingInfo, ...]:
        """Return *program*'s reachable module parameters in closure order."""
        return tuple(
            param
            for module_id in program.closure
            for param in self.module_params.get(module_id, ())
        )


@dataclass(frozen=True, slots=True)
class ArgumentPreflight:
    """Result of ``PipelineDriver.preflight_arguments``.

    ``result``
        The check-only run result: ``ok`` iff the static pipeline succeeded
        and every argument bound and decoded against the program's signature.
    ``executable``
        The lowered program *arguments* were checked against, or ``None``
        when a pass before lowering failed. Hand it back to
        ``PipelineDriver.run_prepared`` as ``executable`` (with this same
        ``arguments``) to execute it without lowering it a second time.
    ``arguments``
        The bound, decoded arguments, ready for ``run_prepared``'s own
        ``arguments`` — one entry per declared parameter, in declaration
        order — or ``()`` when the static pipeline or binding failed.
    ``param_seeds``
        The decoded module-parameter values, ready for ``run_prepared``.
    """

    result: "RunResult"
    executable: "ExecutableProgram | None"
    arguments: "tuple[Value | UseDefault, ...]" = ()
    param_seeds: "Mapping[StaticBindingKey, Value]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedEntry:
    """Result of :meth:`PipelineDriver.parse_entry`.

    The entry source parsed once, ahead of module-graph loading and scope
    resolution. Lets a caller transform the entry AST before
    :meth:`PipelineDriver.prepare_parsed_entry` loads imports and resolves the
    whole program.

    ``program``
        The parsed entry AST, or ``None`` when parsing failed (in which case
        ``diagnostics`` holds the error).
    ``next_id``
        The first node id not yet consumed by this parse — the seed for
        module-graph loading.
    ``spaced_qualifiers``
        Lexical advisories collected while parsing the entry, threaded into
        module-graph loading.
    ``diagnostics``
        A parse failure, or empty on success.
    ``warnings``
        TAB advisories collected while parsing the entry.
    """

    source: str
    entry_path: "Path | None"
    program: "Program | None"
    next_id: int
    spaced_qualifiers: "tuple[SpacedQualifier, ...]"
    diagnostics: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]


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
        declares no extern), keyed by module id. Empty when loading failed.
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


@dataclass(frozen=True, slots=True)
class RunError:
    """Structured representation of an uncaught AgL exception.

    ``type_name`` is the exception's declared type name (e.g. ``"AgentParseError"``).
    ``fields`` is a mapping from field names to JSON-shaped Python values.
    ``line`` is the 1-based source line of the raise site when known; ``None`` when
    the span was not threaded through (e.g. arithmetic errors inside expressions).
    ``col`` is the 1-based source column of the raise site; ``None`` when unknown.
    """

    type_name: str
    fields: dict[str, object]
    line: int | None = None
    col: int | None = None

    def to_message(self) -> str:
        """Render the single-line ``AgL exception: ...`` report for this error."""
        parts: list[str] = [f"AgL exception: {self.type_name}"]
        message = self.fields.get("message")
        if isinstance(message, str) and message:
            parts.append(message)
        if self.line is not None:
            if self.col is not None:
                parts.append(f"at line {self.line}, col {self.col}")
            else:
                parts.append(f"at line {self.line}")
        return ": ".join(parts)


@dataclass(slots=True)
class RunResult:
    """Result of a ``PipelineDriver.run`` call.

    ``ok``
        ``True`` iff there are no error-severity ``diagnostics`` **and** no
        uncaught AgL exception.  ``warnings`` never affect ``ok``.
    ``diagnostics``
        Pre-execution FAILURES only: error-severity items from
        lex/parse/scope/typecheck/matchcompile/program-argument-validation.  Each entry has a
        ``.message`` (str) and a ``.line`` (int, 1-based).  Warnings are a
        SEPARATE channel and NEVER appear here; on a successful run this list is
        empty.
    ``warnings``
        Advisory warning-severity diagnostics (e.g. an unused binding)
        surfaced on EVERY path — success, static failure, program-argument-validation
        failure, and uncaught exception.  Same ``Diagnostic`` type as
        ``diagnostics`` but with ``.severity == "warning"``.  Reported to the
        user but never cause the run to fail (never affect ``ok``).
    ``error``
        The uncaught AgL exception, or ``None``.  Set only when the program
        *started* executing but ended with an unhandled exception (exit code 2
        per the CLI contract).  ``None`` for pre-execution failures and for
        successful runs.
    ``bindings``
        Entry-module public bindings after a successful run (name → Value); a
        scoped binding or agent appears under its full path spelling
        (``A::x``). An explicitly selected synthetic inline ``main`` also
        contributes its direct bindings; an explicit file entry does not.
        Empty for failed runs.
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
    agent_dispatcher : callable or None
        The callable used to dispatch a typed ``Agent`` value for ``ask``.
    shell_exec_timeout : float or None
        Initial idle timeout (in seconds) for the ``std/config::timeout``
        binding used by omitted ``exec`` timeout arguments. ``None`` means no
        timeout; an explicit call argument takes precedence.
    default_call_depth_limit : int or None
        Maximum call depth for recursive functions.  Exceeding
        this limit raises a ``RecursionError`` in the AgL program.  ``None``
        applies the canonical default (``IrInterpreter.DEFAULT_MAX_CALL_DEPTH``).
        Resolved by the caller as ``--max-call-depth`` > ``[exec] max-call-depth``.
    extern_registry : ExternRegistry or None
        Optional shared Python FFI registry. Hosts that run several drivers
        across one program invocation pass the same registry to each so
        companion module imports and Python module globals are shared. Each
        run still creates an interpreter with its own companion runtime state.
    """

    def __init__(
        self,
        *,
        default_strict_json: bool = False,
        agent_dispatcher: AgentFn | None = None,
        session_host: "SessionHost | None" = None,
        shell_exec_timeout: float | None = None,
        default_call_depth_limit: int | None = None,
        extern_registry: "ExternRegistry | None" = None,
    ) -> None:
        self._default_strict_json = default_strict_json
        self._agent_dispatcher = agent_dispatcher
        self._session_host = session_host
        self._shell_exec_timeout = shell_exec_timeout
        self._default_call_depth_limit = (
            default_call_depth_limit
            if default_call_depth_limit is not None
            else IrInterpreter.DEFAULT_MAX_CALL_DEPTH
        )
        self._extern_registry = extern_registry
        # Extra codecs registered by the host (beyond the built-ins).
        self._extra_codecs: dict[str, "OutputCodec"] = {}
        # Cached assembled environment: invariant between registrations, so the
        # REPL's per-entry ``host_environment()`` calls reuse one bundle.  Any
        # ``register_*`` invalidates it.
        self._host_env_cache: HostEnvironment | None = None
        # Only preflight exposes lowered executables for later resumption. Keep
        # their source and capability provenance here rather than in typeless IR.
        self._executable_provenance: dict[int, _ExecutableProvenance] = {}

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

        Returns the value dispatcher, derived ``HostCapabilities``, and merged
        codec tables. The bundle is invariant between codec registrations, so it
        is assembled once and cached.
        """
        if self._host_env_cache is not None:
            return self._host_env_cache
        self._host_env_cache = assemble_host_environment(
            agent_dispatcher=self._agent_dispatcher,
            session_host=self._session_host,
            extra_codecs=self._extra_codecs,
            extern_registry=self._extern_registry,
        )
        return self._host_env_cache

    def _validate_cached_executable(
        self,
        executable: "ExecutableProgram",
        resolved: "ResolvedProgram",
        capabilities: "HostCapabilities",
    ) -> "tuple[ExecutableProgram | None, ModuleId]":
        """Validate a preflight executable and retain its selected module."""
        provenance = self._executable_provenance.get(id(executable))
        if provenance is None:
            raise ArtifactProvenanceError("Cached executable was not produced by this pipeline.")
        if provenance.prepared.resolved is not resolved:
            raise ArtifactProvenanceError(
                "Cached executable does not belong to the prepared source."
            )
        if provenance.capabilities != capabilities:
            return None, provenance.selected_module
        return executable, provenance.selected_module

    def _execute_ir(
        self,
        executable: "ExecutableProgram",
        *,
        host_env: HostEnvironment,
        check_only: bool,
        log_file: "Path | None",
        warnings: list[Diagnostic],
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
        builtin_var_seeds: "Mapping[BuiltinVarKey, Value] | None" = None,
        param_seeds: "Mapping[StaticBindingKey, Value] | None" = None,
        process_environment: "Mapping[str, str] | None" = None,
        program_symbol: "SymbolId | None" = None,
        select_default_program: bool = False,
        arguments: "tuple[Value | UseDefault, ...] | None" = None,
    ) -> RunResult:
        """Run a freshly lowered ``executable`` — the shared tail of the
        shared pipeline tail.

        Materializes host codec contracts, honours the ``check_only`` dry-run
        stop (call-site inventory, no execution), then builds and runs the
        :class:`IrInterpreter`, mapping an uncaught ``AglRaise`` to a failing
        ``RunResult``. All return paths carry *warnings*.

        *arguments* is *program_symbol*'s own bound value-parameter argument
        list, in declaration order. When the caller has already validated it
        against the same executable (:meth:`preflight_arguments`), a length
        or requiredness mismatch here is a caller contract violation, not a
        condition this method re-checks. ``None`` (the default) means no
        argument source was ever consulted: every parameter is derived to its
        own default, and a required parameter with none becomes the same
        pre-execution diagnostic :func:`~agm.agl.runtime.arguments.
        bind_program_arguments` reports for an explicit miss — never an
        interpreter arity crash.
        """
        host_contracts, contract_errors = materialize_ir_contracts(executable, host_env.codecs)
        if contract_errors:
            return RunResult(
                ok=False,
                diagnostics=contract_errors,
                error=None,
                warnings=list(warnings),
            )

        if select_default_program and program_symbol is None:
            entry_programs = tuple(
                symbol
                for symbol, function_id in executable.program_functions.items()
                if executable.functions[function_id].module_id == executable.entry_module
            )
            if len(entry_programs) > 1:
                return RunResult(
                    ok=False,
                    diagnostics=[
                        Diagnostic(message="multiple programs declared; select one", line=1)
                    ],
                    error=None,
                    warnings=list(warnings),
                    bindings={},
                    trace_path=None,
                )
            if len(entry_programs) == 1:
                program_symbol = entry_programs[0]

        # ----------------------------------------------------------------
        # Program entry arguments. ``None`` means the caller consulted no
        # argument source (:meth:`preflight_arguments` was never run for this
        # invocation): derive one from the selected program's own signature,
        # deferring every parameter to its own default. A parameter without
        # one is a pre-execution failure here, before any evaluation, rather
        # than an interpreter call short of arguments.
        # ----------------------------------------------------------------
        if program_symbol is not None and arguments is None:
            from agm.agl.runtime.arguments import default_program_arguments

            arguments, missing_diagnostics = default_program_arguments(
                executable.program_signatures[program_symbol]
            )
            if missing_diagnostics:
                return RunResult(
                    ok=False,
                    diagnostics=list(missing_diagnostics),
                    error=None,
                    warnings=list(warnings),
                    bindings={},
                    trace_path=None,
                )
        elif arguments is None:
            arguments = ()

        # ----------------------------------------------------------------
        # [check_only] --dry-run stop: the full static pipeline, program-argument
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
                trace=trace, policy=host_settings_policy
            )
        else:
            reconfigurer = None

        # ``builtin_host_settings`` is the legacy, engine-key-only API. Keep
        # it for config-host compatibility while exposing module-qualified
        # seeds for every host-backed standard-library binding.
        interpreter_builtin_settings: dict[str | BuiltinVarKey, Value] = {}
        if builtin_host_settings is not None:
            interpreter_builtin_settings.update(builtin_host_settings)
        if builtin_var_seeds is not None:
            # The structured API wins for a ``std/config`` key supplied by
            # both routes: it is the more specific host declaration.
            for key, value in builtin_var_seeds.items():
                interpreter_builtin_settings[key] = value

        try:
            interp = IrInterpreter(
                executable,
                agent_dispatcher=host_env.agent_dispatcher,
                session_host=host_env.session_host,
                strict_json=self._default_strict_json,
                shell_exec_timeout=self._shell_exec_timeout,
                trace=trace,
                max_call_depth=self._default_call_depth_limit,
                host_contracts=host_contracts,
                extern_registry=host_env.extern_registry,
                host_reconfigurer=reconfigurer,
                builtin_host_settings=interpreter_builtin_settings,
                param_seeds=param_seeds,
                process_environment=process_environment,
            )
            entry_bindings = interp.run(program_symbol=program_symbol, arguments=arguments)
        except AglRaise as exc:
            # Uncaught AgL exception (exit code 2 per the CLI contract).
            # ONLY the AgL exception carrier is caught here: an unexpected Python
            # exception is an interpreter bug and must propagate (crash loudly)
            # rather than masquerade as a user-facing pre-execution diagnostic.
            error = exception_value_to_run_error(
                exc.exc,
                nominals=executable.nominals,
                span=exc.span,
                exception_field_encodes=executable.exception_field_encodes,
            )
            # Record the uncaught exception in the trace.
            trace.exception(
                type_name=error.type_name,
                message=str(error.fields.get("message", "")),
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
        except SystemExit as exc:
            trace.run_end(ok=exc.code is None or exc.code == 0)
            raise
        except HostConfigurationError as exc:
            # An invalid startup host value, or an unseeded non-engine binding
            # read during evaluation, is a host-configuration failure (exit 1
            # per the CLI contract), not an uncaught AgL exception. Report it
            # as an ordinary language diagnostic rather than ``result.error``.
            trace.run_end(ok=False)
            return RunResult(
                ok=False,
                diagnostics=[Diagnostic(message=str(exc), line=1)],
                error=None,
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
            bindings=entry_bindings,
            trace_path=trace.path,
        )

    @staticmethod
    def parse_entry(
        entry_source: str,
        *,
        entry_path: "Path | None" = None,
        inline_command: bool = False,
    ) -> ParsedEntry:
        """Parse *entry_source* once, ahead of module-graph loading.

        The first half of :meth:`prepare_program`, split out before module
        loading so a host can transform an entry AST before scope resolution.
        Collects TAB and spaced-qualifier advisories.  Non-raising: an
        ``AglSyntaxError`` is captured into :attr:`ParsedEntry.diagnostics`
        with ``program`` left ``None``.
        *inline_command* applies the ``agm exec -c`` synthetic-entry wrap.
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.modules.loader import EntryParseSyntaxError, parse_entry_module
        from agm.agl.parser import AglSyntaxError

        with tab_warning_collector() as tab_sink:
            try:
                with frontend_recursion_boundary():
                    parsed_module = parse_entry_module(
                        entry_source, entry_path=entry_path, inline_command=inline_command
                    )
            except AglSyntaxError as exc:
                spaced_qualifiers = (
                    exc.spaced_qualifiers if isinstance(exc, EntryParseSyntaxError) else ()
                )
                return ParsedEntry(
                    source=entry_source,
                    entry_path=entry_path,
                    program=None,
                    next_id=0,
                    spaced_qualifiers=spaced_qualifiers,
                    diagnostics=(exc.to_diagnostic(),),
                    warnings=tuple(tab_sink),
                )
            except AglError as exc:
                return ParsedEntry(
                    source=entry_source,
                    entry_path=entry_path,
                    program=None,
                    next_id=0,
                    spaced_qualifiers=(),
                    diagnostics=(exc.to_diagnostic(),),
                    warnings=tuple(tab_sink),
                )
            except Exception as exc:
                return ParsedEntry(
                    source=entry_source,
                    entry_path=entry_path,
                    program=None,
                    next_id=0,
                    spaced_qualifiers=(),
                    diagnostics=(Diagnostic(message=str(exc), line=1),),
                    warnings=tuple(tab_sink),
                )
        program = parsed_module.program
        next_id = parsed_module.next_id

        return ParsedEntry(
            source=entry_source,
            entry_path=entry_path,
            program=program,
            next_id=next_id,
            spaced_qualifiers=parsed_module.spaced_qualifiers,
            diagnostics=(),
            warnings=tuple(tab_sink),
        )

    @staticmethod
    def prepare_parsed_entry(
        parsed: ParsedEntry,
        *,
        roots: "RootSet | None" = None,
        package_roots: "Iterable[PackageInfo]" = (),
        default_stdlib: bool = True,
    ) -> PreparedProgram:
        """Load imports and resolve scope for an already-parsed entry.

        The second half of :meth:`prepare_program`, taking
        :meth:`parse_entry`'s result: drives
        ``load imports → resolve_program``.

        Non-raising in the same way as :meth:`prepare_program`: every load or
        scope failure is captured into :attr:`PreparedProgram.diagnostics`
        rather than raised, with ``resolved`` left ``None``.
        """
        from agm.agl.lexer import tab_warning_collector
        from agm.agl.modules.errors import (
            AmbiguousModule,
            ImportEntryError,
            MissingExternCompanion,
            ModuleNotFound,
            ModulePrefixNotFound,
        )
        from agm.agl.modules.loader import build_repl_graph
        from agm.agl.parser import AglSyntaxError
        from agm.agl.scope import AglScopeError
        from agm.agl.scope.program import resolve_program
        from agm.util.text import normalize_newlines

        entry_source = parsed.source
        entry_path = parsed.entry_path

        if roots is None:
            from pathlib import Path

            from agm.agl.modules.roots import RootSet, assemble_roots
            from agm.config.module_roots import (
                ModuleRootsConfig,
                StdlibResolutionError,
                resolve_lib_root,
                resolve_stdlib_root,
            )

            cwd = Path.cwd()
            entry_directory = entry_path.resolve().parent if entry_path is not None else cwd
            try:
                default_stdlib_root = resolve_stdlib_root(home=Path.home(), anchor=entry_directory)
            except StdlibResolutionError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    RootSet(roots=frozenset()),
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    parsed.warnings,
                )
            roots = assemble_roots(
                invocation_root=entry_directory,
                stdlib_root=default_stdlib_root,
                lib_root=resolve_lib_root(
                    ModuleRootsConfig(lib_root=None, extra=()), home=Path.home()
                ),
                configured=(),
                cli=(),
                cwd=cwd,
                package_roots=package_roots,
            )

        if parsed.program is None:
            return PreparedProgram(
                entry_source, entry_path, roots, None, parsed.diagnostics, parsed.warnings
            )

        with tab_warning_collector() as tab_sink:
            try:
                with frontend_recursion_boundary():
                    graph, next_id, _ = build_repl_graph(
                        parsed.program,
                        parsed.next_id,
                        path=entry_path,
                        cached={},
                        roots=roots,
                        default_stdlib=default_stdlib,
                        spaced_qualifiers=parsed.spaced_qualifiers,
                        default_label="<command>",
                        source_text=normalize_newlines(entry_source),
                    )
            except AglSyntaxError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    (*parsed.warnings, *tab_sink),
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
                    (*parsed.warnings, *tab_sink),
                )
            except AglError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (exc.to_diagnostic(),),
                    (*parsed.warnings, *tab_sink),
                )
            except Exception as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    roots,
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    (*parsed.warnings, *tab_sink),
                )
        warnings: tuple[Diagnostic, ...] = (*parsed.warnings, *tab_sink)

        try:
            with frontend_recursion_boundary():
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

        companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
        return PreparedProgram(
            entry_source,
            entry_path,
            roots,
            resolved,
            (),
            warnings,
            companion_paths,
        )

    @staticmethod
    def prepare_program(
        entry_source: str,
        *,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        package_roots: "Iterable[PackageInfo]" = (),
        default_stdlib: bool = True,
    ) -> PreparedProgram:
        """Load and resolve the program rooted at *entry_source* once.

        Drives ``parse → load imports → resolve_program`` for the entry module
        and every reachable module — a thin wrapper over :meth:`parse_entry`
        followed by :meth:`prepare_parsed_entry`, kept for callers that have
        no use for the split (most of them).

        Non-raising: any load (``ModuleNotFound``, ``AmbiguousModule``,
        ``ModulePrefixNotFound``, ``ImportEntryError``), parse
        (``AglSyntaxError``), or scope (``AglScopeError``) failure is
        captured into :attr:`PreparedProgram.diagnostics` rather than raised,
        with ``resolved`` left ``None``.  TAB advisories are captured
        as warnings via the lex-pass context manager.
        """
        parsed = PipelineDriver.parse_entry(entry_source, entry_path=entry_path)
        return PipelineDriver.prepare_parsed_entry(
            parsed,
            roots=roots,
            package_roots=package_roots,
            default_stdlib=default_stdlib,
        )

    def run(
        self,
        source: str,
        *,
        check_only: bool = False,
        log_file: "Path | None" = None,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        package_roots: "Iterable[PackageInfo]" = (),
        default_stdlib: bool = True,
        builtin_var_seeds: "Mapping[BuiltinVarKey, Value] | None" = None,
        process_environment: "Mapping[str, str] | None" = None,
    ) -> RunResult:
        """Compile and run a program with the standard module roots by default.

        ``builtin_var_seeds`` supplies typed host values by their complete
        ``(ModuleId, scope_path, name)`` identity. It is independent of the legacy
        engine-only ``builtin_host_settings`` accepted by :meth:`run_prepared`.
        """
        return self.run_prepared(
            self.prepare_program(
                source,
                entry_path=entry_path,
                roots=roots,
                package_roots=package_roots,
                default_stdlib=default_stdlib,
            ),
            check_only=check_only,
            log_file=log_file,
            builtin_var_seeds=builtin_var_seeds,
            process_environment=process_environment,
            select_default_program=True,
        )

    def _discover_static(
        self,
        prepared: PreparedProgram,
        *,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> ProgramDiscovery:
        """Run typecheck + match compilation, reusing a supplied artifact.

        The static-pipeline steps :meth:`discover_programs` needs before
        building its own program-declaration products: entry-module
        resolution, typecheck, match compilation, and (when self-validation
        is enabled) artifact-provenance checking against a reused *compiled*
        artifact.

        Returns a :class:`ProgramDiscovery` whose ``programs`` is always
        empty — :meth:`discover_programs` fills it in on success.

        On success both :attr:`~ProgramDiscovery.checked` and
        :attr:`~ProgramDiscovery.compiled` are non-``None``. On failure
        ``compiled`` is always ``None``; ``checked`` is non-``None`` only for
        a match-compile failure (the checked program is still meaningful),
        and ``None`` for every earlier failure, including a missing entry
        module.
        """
        if prepared.resolved is None:
            return ProgramDiscovery(
                programs=(),
                checked=None,
                compiled=None,
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
            return ProgramDiscovery(
                programs=(),
                checked=None,
                compiled=None,
                diagnostics=tc_diagnostics,
                warnings=all_warnings,
            )

        if compiled is None:
            compiled, match_diagnostics = _run_matchcompile_program(
                checked, prepared.resolved.graph, capabilities
            )
            if compiled is None:
                return ProgramDiscovery(
                    programs=(),
                    checked=checked,
                    compiled=None,
                    diagnostics=match_diagnostics,
                    warnings=all_warnings,
                )

        if checked.entry_id not in checked.modules:
            return ProgramDiscovery(
                programs=(),
                checked=None,
                compiled=None,
                diagnostics=(Diagnostic(message="Entry module not found in program", line=1),),
                warnings=all_warnings,
            )

        return ProgramDiscovery(
            programs=(), checked=checked, compiled=compiled, diagnostics=(), warnings=all_warnings
        )

    def discover_programs(
        self,
        prepared: PreparedProgram,
        *,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> ProgramDiscovery:
        """Discover every ``program def`` declaration with its typed parameter signature.

        A supplied artifact is reused; otherwise the successful artifact is
        returned for later lowering by :meth:`run_prepared`. Reports each
        program's own value-parameter signature (``ProgramDeclInfo.parameters``),
        ready for :meth:`preflight_arguments`/:meth:`run_prepared` with
        ``arguments``.
        """
        static = self._discover_static(prepared, compiled=compiled)
        checked = static.checked
        if static.compiled is None or checked is None:
            return static
        assert prepared.resolved is not None
        return replace(
            static,
            programs=_program_decl_infos(checked, prepared.resolved.graph),
            module_params=_module_param_infos(checked),
        )

    def _wire_externs_or_fail(
        self,
        *,
        checked: "CheckedProgram",
        capabilities: "HostCapabilities",
        host_env: HostEnvironment,
        prepared: PreparedProgram,
        module_ids: "set[ModuleId]",
        nominals: "Mapping[NominalId, NominalDescriptor]",
        functions: "Mapping[FunctionId, FunctionDescriptor]",
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
            module_ids=module_ids,
            nominals=nominals,
            functions=functions,
        )
        if extern_diagnostics:
            return on_failure(extern_diagnostics)
        return None

    def run_prepared(
        self,
        prepared: PreparedProgram,
        *,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled: "MatchCompiledProgram | None" = None,
        checked: "CheckedProgram | None" = None,
        executable: "ExecutableProgram | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
        builtin_var_seeds: "Mapping[BuiltinVarKey, Value] | None" = None,
        param_seeds: "Mapping[StaticBindingKey, Value] | None" = None,
        process_environment: "Mapping[str, str] | None" = None,
        program_symbol: "SymbolId | None" = None,
        select_default_program: bool = False,
        arguments: "tuple[Value | UseDefault, ...] | None" = None,
    ) -> RunResult:
        """Execute an already loaded and scoped program without reloading.

        Resumes the pipeline at type checking: ``check_program`` → match
        compilation → ``lower_program`` → ``IrInterpreter``.

        When *prepared* carries a load/scope failure (``resolved is
        None``), its diagnostics are surfaced unchanged and nothing executes.

        ``compiled``
            When the caller has already typechecked and match-compiled the program
            (for example via :meth:`discover_programs`), pass the result
            here to skip those static passes. ``None`` runs them here.

        ``program_symbol``
            A selected linked ``program def`` symbol to invoke after module
            initializers have run, within the interpreter's managed execution
            boundary. ``None`` invokes no declared entry after initialization.

        ``arguments``
            *program_symbol*'s own bound value-parameter argument list, in
            declaration order (:meth:`preflight_arguments`) — this method
            does not validate a supplied *arguments* against the program's
            signature; preflight it first. ``None`` (the default) derives one
            from *program_symbol*'s own signature instead: every parameter
            defers to its own default, and a required parameter with none is
            a clean pre-execution diagnostic. ``()`` explicitly means zero
            arguments, correct only for a parameterless program — passing it
            for a program with parameters is a caller contract violation.

        ``executable``
            When this driver has already lowered this exact program (via
            :meth:`preflight_arguments`), pass the executable here to run it as-is:
            contract materialization and lowering are skipped, so a program is
            lowered only once however many times a host resumes it. A changed
            host capability set invalidates the executable and lowers afresh;
            an executable from another source or driver is rejected. ``None``
            lowers here, before the check-only stop or evaluation.
        """
        result, _executable = self._run_program(
            prepared,
            check_only=check_only,
            log_file=log_file,
            compiled=compiled,
            checked=checked,
            executable=executable,
            host_settings_policy=host_settings_policy,
            builtin_host_settings=builtin_host_settings,
            builtin_var_seeds=builtin_var_seeds,
            param_seeds=param_seeds,
            process_environment=process_environment,
            program_symbol=program_symbol,
            select_default_program=select_default_program,
            arguments=arguments,
        )
        return result

    def check_prepared(
        self,
        prepared: PreparedProgram,
        *,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> RunResult:
        """Run the full static pipeline over *prepared*, with no execution.

        An intent-revealing façade over ``run_prepared(..., check_only=True)``:
        drives typecheck, match compilation, custom-contract payload
        materialization, and lowering, reporting every static failure
        lowering catches (an invalid ``resource``/``resource-dir`` path, an
        unmaterializable output contract), with no execution.

        ``compiled``
            As in :meth:`run_prepared`: a caller-supplied typecheck/match-compile
            artifact to reuse instead of running those passes here.
        """
        return self.run_prepared(prepared, check_only=True, compiled=compiled)

    def _lower_and_record(
        self,
        prepared: PreparedProgram,
        program: ProgramDeclInfo,
        *,
        compiled: "MatchCompiledProgram | None" = None,
    ) -> "tuple[RunResult, ExecutableProgram | None]":
        """Lower *prepared* under ``check_only`` and record the executable's provenance.

        The tail of :meth:`preflight_arguments`: lowers the program, selecting
        *program*'s module, without executing it, then registers the lowered
        executable's cache provenance so a later ``run_prepared(executable=...)``
        call can validate and reuse it instead of lowering a second time.
        """
        capabilities = self.host_environment().capabilities
        result, executable = self._run_program(
            prepared,
            check_only=True,
            compiled=compiled,
            selected_program=program,
        )
        if executable is not None:
            self._executable_provenance[id(executable)] = _ExecutableProvenance(
                executable=executable,
                prepared=prepared,
                capabilities=capabilities,
                selected_module=program.module,
            )
        return result, executable

    def preflight_arguments(
        self,
        prepared: PreparedProgram,
        program: ProgramDeclInfo,
        arguments: "ProgramArguments",
        *,
        compiled: "MatchCompiledProgram | None" = None,
        param_values: "Mapping[StaticBindingKey, object] | None" = None,
    ) -> ArgumentPreflight:
        """Validate program arguments and module parameter values before execution.

        Runs the static pipeline exactly as :meth:`run_prepared` does under
        ``check_only``, then binds and decodes *arguments* against the
        lowered program's signature for *program*
        (:func:`~agm.agl.runtime.arguments.bind_program_arguments_for`). Hand
        the lowered executable and the bound arguments back to
        :meth:`run_prepared` (``executable=``, ``arguments=``) to execute it
        without lowering it a second time.
        """
        from agm.agl.runtime.arguments import bind_param_values, bind_program_arguments_for

        result, executable = self._lower_and_record(prepared, program, compiled=compiled)
        if executable is None or not result.ok:
            return ArgumentPreflight(result=result, executable=executable)

        bound, argument_diagnostics = bind_program_arguments_for(executable, program, arguments)
        param_seeds, param_diagnostics = bind_param_values(executable, param_values or {})
        diagnostics = (*argument_diagnostics, *param_diagnostics)
        if diagnostics:
            return ArgumentPreflight(
                result=RunResult(
                    ok=False,
                    diagnostics=list(diagnostics),
                    error=None,
                    warnings=result.warnings,
                ),
                executable=executable,
            )
        return ArgumentPreflight(
            result=result,
            executable=executable,
            arguments=bound,
            param_seeds=param_seeds,
        )

    def _run_program(
        self,
        prepared: PreparedProgram,
        *,
        check_only: bool = False,
        log_file: "Path | None" = None,
        compiled: "MatchCompiledProgram | None" = None,
        checked: "CheckedProgram | None" = None,
        executable: "ExecutableProgram | None" = None,
        host_settings_policy: "HostSettingsPolicy | None" = None,
        builtin_host_settings: "Mapping[str, Value] | None" = None,
        builtin_var_seeds: "Mapping[BuiltinVarKey, Value] | None" = None,
        param_seeds: "Mapping[StaticBindingKey, Value] | None" = None,
        process_environment: "Mapping[str, str] | None" = None,
        program_symbol: "SymbolId | None" = None,
        select_default_program: bool = False,
        selected_program: ProgramDeclInfo | None = None,
        arguments: "tuple[Value | UseDefault, ...] | None" = None,
    ) -> "tuple[RunResult, ExecutableProgram | None]":
        """Back program execution and parameter preflight with one pipeline body.

        Returns the run result together with the lowered program it ran (the one
        supplied as *executable*, or the one lowered here), or ``None`` when a
        pass before lowering failed.
        """
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
        selected_module = None if selected_program is None else selected_program.module
        if executable is not None:
            executable, selected_module = self._validate_cached_executable(
                executable, resolved, capabilities
            )
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
            compiled, match_diagnostics = _run_matchcompile_program(
                checked, resolved.graph, capabilities
            )
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

        if executable is None:
            from agm.agl.lower import lower_program
            from agm.agl.syntax.resources import ResourceError

            try:
                with frontend_recursion_boundary():
                    executable = lower_program(compiled, contract_payloads=contract_payloads)
            except (NestingTooDeepError, ResourceError) as exc:
                diagnostic = (
                    diagnostic_from_span(str(exc), exc.span)
                    if exc.span is not None
                    else Diagnostic(message=str(exc), line=1)
                )
                return (
                    RunResult(
                        ok=False,
                        diagnostics=[diagnostic],
                        error=None,
                        warnings=list(warnings),
                    ),
                    None,
                )

        if selected_module is not None:
            executable = _select_program_inventory(executable, resolved.graph, selected_module)

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
                module_ids=set(executable.modules),
                nominals=executable.nominals,
                functions=executable.functions,
                on_failure=lambda extern_diagnostics: RunResult(
                    ok=False,
                    diagnostics=extern_diagnostics,
                    error=None,
                    warnings=warnings,
                ),
            )
            if run_failure is not None:
                return run_failure, None

        return (
            self._execute_ir(
                executable,
                host_env=host_env,
                check_only=check_only,
                log_file=log_file,
                warnings=warnings,
                host_settings_policy=host_settings_policy,
                builtin_host_settings=builtin_host_settings,
                builtin_var_seeds=builtin_var_seeds,
                param_seeds=param_seeds,
                process_environment=process_environment,
                program_symbol=program_symbol,
                select_default_program=select_default_program,
                arguments=arguments,
            ),
            executable,
        )

    @property
    def default_call_depth_limit(self) -> int:
        """Maximum call depth for recursive functions."""
        return self._default_call_depth_limit

    def reset_extern_registry(self) -> None:
        """Replace the cached extern registry with a fresh, empty one.

        Called by ``ReplSession.reset()`` so a session's extern state is
        discarded like every other session-scoped binding: after a reset, a
        library module's companion resolves and imports again as though the
        session were new. The rest of the assembled host environment is left
        untouched — only the extern registry is replaced. It is a no-op before
        the environment has ever been assembled.
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


def _reachable_modules(
    module_id: "ModuleId", adjacency: "Mapping[ModuleId, tuple[ModuleId, ...]]"
) -> tuple[ModuleId, ...]:
    """Return a module and its dependencies in *adjacency* order."""
    reachable: list[ModuleId] = []
    seen: set[ModuleId] = set()
    pending = [module_id]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        reachable.append(current)
        pending.extend(reversed(adjacency[current]))
    return tuple(reachable)


def _select_program_inventory(
    executable: "ExecutableProgram", graph: "ModuleGraph", module_id: "ModuleId"
) -> "ExecutableProgram":
    """Restrict a selected program to its graph-reachable runtime modules and source inventory."""
    source_reachable = frozenset(graph.source_reachable_modules(module_id))
    runtime_reachable = frozenset(_reachable_modules(module_id, graph.adjacency))
    return replace(
        executable,
        entry_module=module_id,
        modules={
            mid: module for mid, module in executable.modules.items() if mid in runtime_reachable
        },
        dry_run_inventory=tuple(
            call_site
            for call_site in executable.dry_run_inventory
            if call_site.module in source_reachable
        ),
        param_bindings={
            key: symbol
            for key, symbol in executable.param_bindings.items()
            if key[0] in source_reachable
        },
        param_decoders={
            key: decoder
            for key, decoder in executable.param_decoders.items()
            if key[0] in source_reachable
        },
        param_spans={
            key: span for key, span in executable.param_spans.items() if key[0] in source_reachable
        },
    )


def _program_decl_infos(
    checked: "CheckedProgram", graph: "ModuleGraph"
) -> tuple[ProgramDeclInfo, ...]:
    """Return every ``program def`` declaration's info, in stable sorted order.

    Pairs each ``program def`` the shared
    :func:`~agm.agl.typecheck.program.program_funcdefs` walk yields with its
    own typed value-parameter signature. Used by
    :meth:`PipelineDriver.discover_programs`.
    """
    from agm.agl.typecheck.program import program_funcdefs

    program_infos = [
        ProgramDeclInfo(
            module=module_id,
            scope_path=tuple(segment.name for segment in item.scope_path),
            name=item.name,
            node_id=item.node_id,
            span=item.span,
            parameters=_program_param_infos(checked, module_id, item),
            is_entry=module_id == checked.entry_id,
            doc=checked_module.resolved.attributes.docs.get(item.node_id),
            closure=graph.source_reachable_modules(module_id),
        )
        for module_id, checked_module, item in program_funcdefs(checked.modules)
    ]
    program_infos.sort(
        key=lambda info: (
            not info.is_entry,
            info.module.path_str(),
            info.declaration_path,
        )
    )
    return tuple(program_infos)


def _module_param_infos(
    checked: "CheckedProgram",
) -> dict["ModuleId", tuple[ParamBindingInfo, ...]]:
    """Return each checked module's marked static bindings in source order."""
    from agm.agl.syntax.nodes import LetDecl, VarDecl, simple_let_pattern_name, static_items

    module_params: dict[ModuleId, tuple[ParamBindingInfo, ...]] = {}
    for module_id, checked_module in checked.modules.items():
        attributes = checked_module.resolved.attributes
        params: list[ParamBindingInfo] = []
        for item in static_items(checked_module.resolved.program.body.items):
            name: str | None
            if isinstance(item, VarDecl):
                binding_node_id = item.node_id
                name = item.name
                type_ann = item.type_ann
            elif isinstance(item, LetDecl):
                binding_node_id = item.pattern.node_id
                name = simple_let_pattern_name(item.pattern)
                type_ann = item.type_ann
            else:
                continue
            cli = attributes.params.get(binding_node_id)
            if cli is None:
                continue
            assert name is not None
            binding_type = checked_module.type_env.get_binding_type(binding_node_id)
            assert binding_type is not None
            params.append(
                ParamBindingInfo(
                    module=module_id,
                    scope_path=tuple(segment.name for segment in item.scope_path),
                    name=name,
                    node_id=binding_node_id,
                    span=item.span,
                    type=binding_type,
                    mutable=isinstance(item, VarDecl),
                    cli=cli,
                    doc=attributes.docs.get(item.node_id),
                    is_path=_annotates_path(
                        checked,
                        module_id,
                        tuple(segment.name for segment in item.scope_path),
                        type_ann,
                        binding_type,
                    ),
                )
            )
        module_params[module_id] = tuple(params)
    return module_params


def _program_param_infos(
    checked: "CheckedProgram", module_id: "ModuleId", funcdef: "FuncDef"
) -> tuple[ProgramParamInfo, ...]:
    """Return *funcdef*'s checked parameter signature as host-facing info.

    Pairs each AST ``Param`` (for its declaration span, annotation, and the
    option presentation scope recognized for it) with the checker's
    ``ParamSpec`` (for its zone, type, and default) positionally — the two
    describe the same parameter list, in the same declaration order.
    """
    checked_module = checked.modules[module_id]
    signature = checked_module.type_env.get_function_signature_by_node_id(funcdef.node_id)
    assert signature is not None, (
        f"compiler bug: program {funcdef.name!r} has no recorded function signature"
    )
    scope_path = tuple(segment.name for segment in funcdef.scope_path)
    return tuple(
        ProgramParamInfo(
            name=param_spec.name,
            kind=param_spec.kind,
            type=param_spec.type,
            has_default=param_spec.has_default,
            span=ast_param.span,
            cli=checked_module.resolved.attributes.program_options[ast_param.node_id],
            is_path=_annotates_path(
                checked, module_id, scope_path, ast_param.type_expr, param_spec.type
            ),
        )
        for ast_param, param_spec in zip(funcdef.params, signature.params, strict=True)
    )


def _annotates_path(
    checked: "CheckedProgram",
    module_id: "ModuleId",
    scope_path: tuple[str, ...],
    type_expr: "TypeExpr | None",
    resolved: "Type",
) -> bool:
    """Return whether *type_expr*, checked as *resolved*, spells the builtin ``path``.

    Resolution erases transparent aliases, so the annotation's own
    declarations are followed instead: non-generic aliases down to the builtin
    alias, or to the reserved ``path`` no declaration reaches, optionally
    wrapped in the standard ``Option``.
    """
    from agm.agl.semantics.types import PATH_TYPE_NAME, is_standard_option_enum
    from agm.agl.syntax.nodes import TypeAlias, static_type_items
    from agm.agl.syntax.types import AppliedT, NameT

    if not isinstance(type_expr, (NameT, AppliedT)):
        return False
    env = checked.modules[module_id].type_env
    with env.type_scope(scope_path):
        key = env.type_name_declaration(type_expr)
    if key is None:
        return isinstance(type_expr, NameT) and type_expr.name == PATH_TYPE_NAME
    decl_module, decl_path, decl_name = key
    alias = next(
        (
            item
            for item in static_type_items(checked.modules[decl_module].resolved.program.body.items)
            if isinstance(item, TypeAlias)
            and item.name == decl_name
            and tuple(segment.name for segment in item.scope_path) == decl_path
        ),
        None,
    )
    if alias is not None:
        if alias.is_builtin:
            return alias.name == PATH_TYPE_NAME
        return not alias.type_params and _annotates_path(
            checked, decl_module, decl_path, alias.type_expr, resolved
        )
    return (
        isinstance(type_expr, AppliedT)
        and is_standard_option_enum(resolved)
        and len(type_expr.args) == len(resolved.type_args) == 1
        and _annotates_path(
            checked, module_id, scope_path, type_expr.args[0], resolved.type_args[0]
        )
    )


def _check_artifact_provenance(
    *,
    prepared_entry: "ModuleId",
    prepared_modules: "Mapping[ModuleId, ModuleResolution]",
    compiled_entry: "ModuleId",
    compiled_modules: "Mapping[ModuleId, ModuleResolution]",
) -> None:
    """Assert a cached artifact wraps the exact prepared resolutions.

    Source identity is the provenance contract: structurally equal resolutions
    prepared in separate passes are not interchangeable compiler inputs. The
    check anchors on the resolution object itself, which typecheck leaves
    unchanged — it records final pattern-slot meanings in checker-owned maps —
    so a compiled artifact must wrap the very object that was prepared, not
    merely one sharing its parsed program. Call sites guard this check with
    :func:`self_validation_enabled` — building its module mappings costs more
    than the production path should ever pay for an invariant it cannot
    violate.
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
        with frontend_recursion_boundary():
            checked = check_program(resolved, capabilities)
    except AglError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        diagnostic = Diagnostic(message=f"Type error: {exc}", line=1)
        return None, (diagnostic,)
    return checked, ()


def _run_matchcompile_program(
    checked: "CheckedProgram",
    graph: "ModuleGraph",
    capabilities: "HostCapabilities",
) -> "tuple[MatchCompiledProgram | None, tuple[Diagnostic, ...]]":
    """Run program-level match compilation without raising.

    The sites an earlier compilation of the same modules left behind are offered
    per module; each is carried over only while it still holds the very checked
    module this program carries.
    """
    from agm.agl.matchcompile import (
        MatchCompiledProgram,
        cached_module_sites,
        compile_program_matches,
        diagnostics_from_match_issues,
    )

    retainable = retained_module_sources(graph)
    try:
        result = compile_program_matches(checked, retained_match_sites(retainable, capabilities))
        if result.compiled is None:
            return None, diagnostics_from_match_issues(result.issues)
        if not isinstance(result.compiled, MatchCompiledProgram):
            raise TypeError("program match compilation returned a module artifact")
    except Exception as exc:
        return None, (Diagnostic(message=f"Match compilation error: {exc}", line=1),)
    retain_match_sites(retainable, capabilities, cached_module_sites(result.compiled))
    return result.compiled, ()


def _materialize_custom_contract_payloads(
    specs: "Mapping[int, OutputContractSpec]",
    codecs: "Mapping[str, OutputCodec]",
    type_table: "TypeTable",
) -> tuple[dict[int, "ContractPayload"], list[Diagnostic]]:
    """Run custom codec contract hooks before lowering and keep only typeless data."""
    from agm.agl.ir.contracts import ContractPayload
    from agm.agl.runtime.codec import BUILTIN_CODEC_NAMES
    from agm.agl.runtime.contract import materialize_contract
    from agm.agl.semantics.types import UnitType

    payloads: dict[int, ContractPayload] = {}
    errors: list[Diagnostic] = []
    for node_id, spec in specs.items():
        if spec.codec_name in BUILTIN_CODEC_NAMES or isinstance(spec.target_type, UnitType):
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
    module_ids: "set[ModuleId] | None" = None,
) -> list[tuple["ModuleId", str]]:
    """Return ``(module_id, companion_name)`` for every declared extern.

    An extern resolves in its declaring module's companion under the name
    scope recorded for it — its ``@extern-name`` argument, or its declared
    member name. Scope rejects two externs of one module claiming the same
    companion name, so this list stays one-to-one with companion callables.
    """
    from agm.agl.syntax.nodes import FuncDef, ScopeRegion

    def collect(items: tuple[object, ...]) -> list[FuncDef]:
        declarations: list[FuncDef] = []
        for item in items:
            if isinstance(item, ScopeRegion):
                declarations.extend(collect(item.items))
            elif isinstance(item, FuncDef) and item.is_extern:
                declarations.append(item)
        return declarations

    declarations = [
        (mid, mod.resolved.attributes.extern_names[funcdef.node_id])
        for mid, mod in checked.modules.items()
        if module_ids is None or mid in module_ids
        for funcdef in collect(mod.resolved.program.body.items)
    ]
    declarations.sort(key=_extern_declaration_sort_key)
    return declarations


def _wire_extern_registry(
    *,
    checked: "CheckedProgram",
    capabilities: "HostCapabilities",
    registry: "ExternRegistry",
    companion_paths: "Mapping[ModuleId, Path | None]",
    module_ids: "set[ModuleId] | None" = None,
    nominals: "Mapping[NominalId, NominalDescriptor] | None" = None,
    functions: "Mapping[FunctionId, FunctionDescriptor] | None" = None,
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
    included_modules = set(companion_paths) if module_ids is None else module_ids
    if not any(companion_paths.get(mid) is not None for mid in included_modules):
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
    declarations = _extern_declarations(checked, module_ids)
    if nominals is not None:
        registry.set_nominals(dict(nominals), functions=functions)

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


def assemble_host_environment(
    *,
    agent_dispatcher: AgentFn | None,
    session_host: "SessionHost | None",
    extra_codecs: dict[str, "OutputCodec"],
    extern_registry: "ExternRegistry | None" = None,
) -> HostEnvironment:
    """Assemble the shared host runtime environment from registrations.

    Builds the merged codec table and the derived ``HostCapabilities`` exactly
    as ``PipelineDriver.run`` does inline. Used by both ``run`` and
    ``ReplSession`` so codec capabilities have one source of truth.
    """
    from agm.agl.capabilities import HostCapabilities
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

    capabilities = HostCapabilities(
        supports_shell_exec=True,
        supports_extern=True,
        codec_kinds={name: codec.supported_kinds for name, codec in all_codecs.items()},
    )
    return HostEnvironment(
        agent_dispatcher=agent_dispatcher,
        session_host=session_host,
        capabilities=capabilities,
        codecs=all_codecs,
        extern_registry=extern_registry if extern_registry is not None else ExternRegistry(),
    )


def exception_value_to_run_error(
    exc: "ExceptionValue",
    *,
    nominals: "Mapping[NominalId, NominalDescriptor]",
    span: "object" = None,  # SourceSpan | None — avoids import cycle
    exception_field_encodes: "Mapping[NominalId, tuple[ExceptionFieldEncode, ...]] | None" = None,
) -> RunError:
    """Convert an ``ExceptionValue`` to a ``RunError`` for ``RunResult``.

    Every field is reported under its effective JSON name (``@json-name`` ??
    ``@name`` ?? declared), from ``exception_field_encodes`` — including a
    field with no JSON form, so no two fields can collide on a shared
    fallback declared key. A field with an encode plan is converted through
    it, matching ``e as json``; one without (not JSON-convertible) goes
    through the shared value-directed serializer instead. Both preserve
    ``Decimal`` exactness (never routed through binary ``float``). When
    ``exception_field_encodes`` has no entry for the exception (e.g. no
    lowering data), every field falls back to its declared name. This runs
    while reporting an error already in flight, so a field that is
    itself cyclic (including one closed through mutable record fields), or
    a field of a kind with no JSON representation (``unit``, ``agent``,
    ``constructor``, ``function``, ``iterator`` — legal on an exception field
    even though a cast to ``json`` of such a type is statically rejected),
    must not raise and mask the real error — that one field is reported as a
    marker instead.

    ``nominals`` resolves *exc*'s display spelling for ``RunError.type_name``
    from the running program's own descriptor table.

    *span* is the optional raise-site source span threaded from ``AglRaise``;
    when present, ``RunError.line`` and ``RunError.col`` are populated from it
    so the CLI can include the source location in its exit-2 error output.
    """
    from agm.agl.ir.ids import Location
    from agm.agl.runtime.serialize import (
        AglNonDataValue,
        degraded_marker,
        encode_value,
        value_to_json_obj,
    )
    from agm.agl.semantics.cycles import AglCyclicValue
    from agm.agl.syntax.spans import SourceSpan

    encodes = {
        encode.field_name: encode
        for encode in (
            () if exception_field_encodes is None else exception_field_encodes.get(exc.nominal, ())
        )
    }
    fields: dict[str, object] = {}
    for k, v in exc.fields.items():
        encode = encodes.get(k)
        # No entry (no lowering data for this exception) falls back to the
        # declared name; an entry always carries its effective JSON name,
        # whether or not it carries a plan.
        json_key = encode.json_name if encode is not None else k
        try:
            fields[json_key] = (
                encode_value(encode.plan, v)
                if encode is not None and encode.plan is not None
                else value_to_json_obj(v)
            )
        except (AglCyclicValue, AglNonDataValue) as field_exc:
            fields[json_key] = degraded_marker(field_exc)
    line: int | None = None
    col: int | None = None
    if isinstance(span, (SourceSpan, Location)):
        line = span.start_line
        col = span.start_col
    return RunError(type_name=nominals[exc.nominal].display_name, fields=fields, line=line, col=col)


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
