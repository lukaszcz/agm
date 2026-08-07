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
from agm.agl.eval.ir_interpreter import HostConfigurationError, IrInterpreter
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.params import _materialize_ir_contracts, _prepare_ir_params
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
    from agm.agl.ir.ids import NominalId
    from agm.agl.ir.program import ExecutableProgram, NominalDescriptor
    from agm.agl.matchcompile import MatchCompiledProgram
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule, ModuleGraph
    from agm.agl.modules.roots import RootSet
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.runtime.host_settings import HostSettingsPolicy
    from agm.agl.scope.program import ResolvedProgram
    from agm.agl.scope.symbols import ModuleResolution
    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.values import ExceptionValue, Value
    from agm.agl.setting_overrides import SettingOverride
    from agm.agl.syntax.advisories import SpacedQualifier
    from agm.agl.syntax.nodes import Program
    from agm.agl.typecheck.env import OutputContractSpec
    from agm.agl.typecheck.program import CheckedProgram

_ResultT = TypeVar("_ResultT")


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
class ParsedEntry:
    """Result of :meth:`PipelineDriver.parse_entry`.

    The entry source parsed once, ahead of module-graph loading and scope
    resolution. Lets a caller read the declared program name and choose
    engine-setting overrides before :meth:`PipelineDriver.prepare_parsed_entry`
    loads imports and resolves the whole program.

    ``program``
        The parsed entry AST, or ``None`` when parsing failed (in which case
        ``diagnostics`` holds the error).
    ``next_id``
        The first node id not yet consumed by this parse — the seed for
        module-graph loading.
    ``program_name``
        The declared ``program NAME``, read directly from the parsed AST
        without requiring scope resolution; ``None`` when the entry declares
        none (or parsing failed).
    ``spaced_qualifiers``
        Lexical advisories collected while parsing the entry, threaded into
        module-graph loading exactly as :func:`~agm.agl.modules.loader.load_graph`
        does.
    ``diagnostics``
        A parse failure, or empty on success.
    ``warnings``
        TAB advisories collected while parsing the entry.
    """

    source: str
    entry_path: "Path | None"
    program: "Program | None"
    next_id: int
    program_name: str | None
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
        Advisory warning-severity diagnostics (e.g. an unused binding)
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
        Entry-module public bindings after a successful run (name → Value); a
        scoped binding or agent appears under its full path spelling
        (``A::x``).  Empty for failed runs.
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
    agent_dispatcher : callable or None
        The callable used to dispatch a typed ``Agent`` value for ``ask``.
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
        agent_dispatcher: AgentFn | None = None,
        shell_exec_timeout: float | None = None,
        default_call_depth_limit: int | None = None,
        extern_registry: "ExternRegistry | None" = None,
    ) -> None:
        self._default_strict_json = default_strict_json
        self._default_loop_limit = default_loop_limit
        self._agent_dispatcher = agent_dispatcher
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
                trace=trace, policy=host_settings_policy
            )
        else:
            reconfigurer = None

        try:
            interp = IrInterpreter(
                executable,
                agent_dispatcher=host_env.agent_dispatcher,
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
            entry_bindings = interp.run()
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
        except HostConfigurationError as exc:
            # The materialized ``default-agent`` value cannot be dispatched: a
            # pre-execution host-configuration failure (exit 1 per the CLI
            # contract), not an uncaught AgL exception — nothing has executed
            # yet, so it is reported as an ordinary diagnostic rather than
            # ``result.error``.
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
    ) -> ParsedEntry:
        """Parse *entry_source* once, ahead of module-graph loading.

        The first half of :meth:`prepare_program`, split out so a caller can
        read :attr:`ParsedEntry.program_name` — and so a host driving a
        second program-point step (e.g. choosing engine-setting overrides)
        never re-parses the entry to get there.  Collects TAB and
        spaced-qualifier advisories exactly as
        :func:`~agm.agl.modules.loader.load_graph` does for its own entry
        parse.  Non-raising: an ``AglSyntaxError`` is captured into
        :attr:`ParsedEntry.diagnostics` with ``program`` left ``None``.
        """
        from agm.agl.lexer import spaced_qualifier_collector, tab_warning_collector
        from agm.agl.parser import AglSyntaxError
        from agm.agl.parser.parser import parse_program_seeded
        from agm.agl.syntax.nodes import ProgramDecl
        from agm.agl.syntax.spans import SourceId

        canonical_entry_path = entry_path.resolve() if entry_path is not None else None
        label = str(canonical_entry_path) if canonical_entry_path is not None else "<command>"
        entry_source_id = SourceId(label=label)

        with tab_warning_collector() as tab_sink, spaced_qualifier_collector() as spaced_sink:
            try:
                program, next_id = parse_program_seeded(
                    entry_source, start_id=0, source=entry_source_id
                )
            except AglSyntaxError as exc:
                return ParsedEntry(
                    source=entry_source,
                    entry_path=entry_path,
                    program=None,
                    next_id=0,
                    program_name=None,
                    spaced_qualifiers=tuple(spaced_sink),
                    diagnostics=(exc.to_diagnostic(),),
                    warnings=tuple(tab_sink),
                )

        program_name: str | None = None
        for item in program.body.items:
            if isinstance(item, ProgramDecl):
                program_name = item.name
                break

        return ParsedEntry(
            source=entry_source,
            entry_path=entry_path,
            program=program,
            next_id=next_id,
            program_name=program_name,
            spaced_qualifiers=tuple(spaced_sink),
            diagnostics=(),
            warnings=tuple(tab_sink),
        )

    @staticmethod
    def prepare_parsed_entry(
        parsed: ParsedEntry,
        *,
        roots: "RootSet | None" = None,
        default_stdlib: bool = True,
        setting_overrides: "Mapping[str, SettingOverride] | None" = None,
    ) -> PreparedProgram:
        """Load imports and resolve scope for an already-parsed entry.

        The second half of :meth:`prepare_program`, taking
        :meth:`parse_entry`'s result: drives
        ``load imports → apply setting_overrides → resolve_program``.

        ``setting_overrides``, when given, splices each named engine key's
        AgL source text in as its ``std/config`` ``builtin var`` declaration's
        default *after* the module graph is loaded but *before* scope
        resolution — so the override is resolved, type-checked, and
        constant-checked by this same pass, never by a second compilation.  A
        rejected override (unparseable source, not exactly one expression, an
        unknown engine key, or a graph with no loaded ``std/config``) is
        captured as a diagnostic naming the override's origin; a type or
        constant-expression violation surfaces later, from the ordinary
        ``builtin var`` checking in :meth:`run_prepared`/:meth:`discover_params`.

        Non-raising in the same way as :meth:`prepare_program`: every load,
        override, or scope failure is captured into
        :attr:`PreparedProgram.diagnostics` rather than raised, with
        ``resolved`` left ``None``.
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
                StaleStdlibError,
                resolve_lib_root,
                resolve_stdlib_root,
            )

            cwd = Path.cwd()
            try:
                default_stdlib_root = resolve_stdlib_root(home=Path.home())
            except StaleStdlibError as exc:
                return PreparedProgram(
                    entry_source,
                    entry_path,
                    RootSet(roots=frozenset()),
                    None,
                    (Diagnostic(message=str(exc), line=1),),
                    parsed.warnings,
                )
            roots = assemble_roots(
                invocation_root=entry_path.resolve().parent if entry_path is not None else cwd,
                stdlib_root=default_stdlib_root,
                lib_root=resolve_lib_root(
                    ModuleRootsConfig(lib_root=None, extra=()), home=Path.home()
                ),
                configured=(),
                cli=(),
                cwd=cwd,
            )

        if parsed.program is None:
            return PreparedProgram(
                entry_source, entry_path, roots, None, parsed.diagnostics, parsed.warnings
            )

        with tab_warning_collector() as tab_sink:
            try:
                graph, next_id, newly_loaded_modules = build_repl_graph(
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

        if setting_overrides:
            # A one-shot compile: whether or not ``std/config`` is in this
            # graph, this is the only chance to apply/validate the overrides,
            # so a ``required`` override (e.g. ``--agent``) must be validated
            # even when ``std/config`` never loads (``validate_when_absent=True``).
            graph, next_id, override_diagnostics, _newly_loaded_modules = apply_setting_overrides(
                graph,
                next_id,
                setting_overrides,
                newly_loaded_modules=newly_loaded_modules,
                validate_when_absent=True,
            )
            if override_diagnostics:
                return PreparedProgram(
                    entry_source, entry_path, roots, None, tuple(override_diagnostics), warnings
                )

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

        companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
        return PreparedProgram(
            entry_source, entry_path, roots, resolved, (), warnings, companion_paths
        )

    @staticmethod
    def prepare_program(
        entry_source: str,
        *,
        entry_path: "Path | None" = None,
        roots: "RootSet | None" = None,
        default_stdlib: bool = True,
        setting_overrides: "Mapping[str, SettingOverride] | None" = None,
    ) -> PreparedProgram:
        """Load and resolve the program rooted at *entry_source* once.

        Drives ``parse → load imports → resolve_program`` for the entry module
        and every reachable module — a thin wrapper over :meth:`parse_entry`
        followed by :meth:`prepare_parsed_entry`, kept for callers that have
        no use for the split (most of them).

        ``setting_overrides`` maps an engine key (e.g. ``"default-agent"``) to
        a :class:`~agm.agl.setting_overrides.SettingOverride` supplying its
        default as host-provided AgL source text; see
        :meth:`prepare_parsed_entry` for how it is applied and diagnosed.

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
            default_stdlib=default_stdlib,
            setting_overrides=setting_overrides,
        )

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
        from agm.agl.syntax.nodes import ParamDecl, scoped_public_name, static_items

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
        for item in static_items(entry_cm.resolved.program.body.items):
            if isinstance(item, ParamDecl):
                param_type = entry_cm.type_env.get_binding_type(item.node_id)
                assert param_type is not None, (
                    f"Param {item.name!r} has no recorded binding type; checker invariant violated."
                )
                external_name = scoped_public_name(item.scope_path, item.name)
                infos.append(
                    ParamDeclInfo(
                        name=external_name,
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
        nominals: "Mapping[NominalId, NominalDescriptor]",
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
            nominals=nominals,
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
        compilation → ``lower_program`` → ``IrInterpreter``.

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

        if executable is None:
            from agm.agl.lower import lower_program

            executable = lower_program(compiled, contract_payloads=contract_payloads)

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
                nominals=executable.nominals,
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


def apply_setting_overrides(
    graph: "ModuleGraph",
    next_id: int,
    overrides: "Mapping[str, SettingOverride]",
    *,
    newly_loaded_modules: "Mapping[ModuleId, LoadedModule]",
    validate_when_absent: bool,
) -> "tuple[ModuleGraph, int, list[Diagnostic], dict[ModuleId, LoadedModule]]":
    """Own the splice-once apply condition and module-cache reconciliation.

    The public seam both one-shot hosts (``PipelineDriver.prepare_parsed_entry``)
    and incremental ones (``EntryPipeline.load_and_check_program``, shared by
    the REPL's ``open()`` and ``eval_entry``) call, so neither has to
    rediscover *when* ``_apply_setting_overrides`` may run or how to keep a
    caller's own module cache in sync with the spliced result.

    *newly_loaded_modules* is the ``build_repl_graph``/``load_and_check_program``
    "loaded during this call" dict (not the caller's whole cache): it decides
    which of three states *graph* is in for ``std/config``, keyed off
    :data:`~agm.agl.modules.ids.STD_CONFIG_ID`:

    - Already spliced by an earlier call this session (present in
      ``graph.modules`` but not freshly loaded this round) — a no-op, since
      splicing a second time would both waste work and mint a new, unstable
      declaration identity for the same ``builtin var``.
    - Freshly loaded this round — splice via :func:`_apply_setting_overrides`,
      then return *newly_loaded_modules* updated with the spliced module, so
      whichever cache the caller promotes it into (the REPL's
      ``_loaded_lib_modules``) holds the SPLICED module, not the pre-splice one.
    - Never loaded at all — nothing to splice into. *validate_when_absent*
      decides whether this is reported now: ``True`` (every one-shot host,
      and the REPL's ``open()``, which validates a CLI-required override up
      front even without ``std/config``) still runs
      :func:`_apply_setting_overrides` so a ``required`` override's
      diagnostic surfaces as early as the host can report it; ``False`` (an
      ordinary REPL entry) defers entirely, leaving even a ``required``
      override unvalidated until whichever later entry, if any, first loads
      ``std/config``.

    Returns *overrides* unchanged (empty diagnostics, *newly_loaded_modules*
    as given) when *overrides* is empty, so every caller can call this
    unconditionally.
    """
    from agm.agl.modules.ids import STD_CONFIG_ID

    if not overrides:
        return graph, next_id, [], dict(newly_loaded_modules)

    freshly_loaded = STD_CONFIG_ID in newly_loaded_modules
    already_loaded = STD_CONFIG_ID in graph.modules

    if not freshly_loaded and already_loaded:
        return graph, next_id, [], dict(newly_loaded_modules)
    if not freshly_loaded and not already_loaded and not validate_when_absent:
        return graph, next_id, [], dict(newly_loaded_modules)

    graph, next_id, diagnostics = _apply_setting_overrides(graph, next_id, overrides)
    if diagnostics or not freshly_loaded:
        return graph, next_id, diagnostics, dict(newly_loaded_modules)
    reconciled = {**newly_loaded_modules, STD_CONFIG_ID: graph.modules[STD_CONFIG_ID]}
    return graph, next_id, [], reconciled


def _apply_setting_overrides(
    graph: "ModuleGraph",
    next_id: int,
    overrides: "Mapping[str, SettingOverride]",
) -> "tuple[ModuleGraph, int, list[Diagnostic]]":
    """Splice each override's parsed expression in as its engine key's default.

    Runs after the module graph is loaded and before scope resolution, so
    every spliced expression is resolved, type-checked, and constant-checked
    by the ordinary ``std/config`` ``builtin var`` machinery
    (``typecheck.checker._check_builtin_var``) exactly as if it had been
    written in ``std/config.agl`` itself — no separate compilation. Node ids
    for the parsed override expressions are seeded from *next_id*, the first
    id not yet used anywhere in *graph*, so they stay disjoint from every
    loaded module.  The returned ``int`` is the first id still unused after
    every override's expression was parsed — a one-shot batch caller (e.g.
    ``prepare_parsed_entry``) has no further use for it, but an incremental
    host that keeps minting node ids afterward (the REPL, which caches and
    reuses the spliced module across later entries) must continue from it
    rather than from the pre-splice *next_id*, to keep every later entry's
    ids disjoint from the ones spliced here.

    Diagnostics — never exceptions — are returned for: unparseable override
    source, an override that is not exactly one expression, an engine key no
    loaded ``builtin var`` declares, and a graph with no loaded ``std/config``
    module (e.g. ``default_stdlib=False`` with no explicit import of it) when
    the override is :attr:`~agm.agl.setting_overrides.SettingOverride.required`
    — a non-``required`` override is skipped silently in that case instead,
    since it is ambient configuration rather than a request the host must
    honor. Each diagnostic names the offending override's ``origin``.
    Overrides are processed in sorted key order for deterministic diagnostics.

    Callers reach this only through :func:`apply_setting_overrides`, which
    decides *when* it may run; this function itself has no opinion on that.
    """
    from dataclasses import replace as dc_replace

    from agm.agl.modules.ids import STD_CONFIG_ID
    from agm.agl.parser import AglSyntaxError
    from agm.agl.parser.parser import parse_program_seeded
    from agm.agl.syntax.nodes import BuiltinVarDecl, Expr
    from agm.agl.syntax.spans import SourceId

    diagnostics: list[Diagnostic] = []
    modules = dict(graph.modules)
    std_config = modules.get(STD_CONFIG_ID)

    for key in sorted(overrides):
        override = overrides[key]
        if std_config is None and not override.required:
            # Ambient configuration, not a request: inert when std/config
            # never loads, so it is skipped without even being parsed.
            continue
        try:
            override_program, next_id = parse_program_seeded(
                override.source, start_id=next_id, source=SourceId(label=override.origin)
            )
        except AglSyntaxError as exc:
            base = exc.to_diagnostic()
            diagnostics.append(
                dc_replace(
                    base,
                    message=f"{override.origin}: invalid AgL expression: {base.message}",
                )
            )
            continue

        items = override_program.body.items
        if len(items) != 1 or not isinstance(items[0], Expr):
            diagnostics.append(
                diagnostic_from_span(
                    f"{override.origin}: expected exactly one AgL expression, "
                    f"got {override.source!r}",
                    override_program.span,
                )
            )
            continue
        value_expr = items[0]

        if std_config is None:
            diagnostics.append(
                diagnostic_from_span(
                    f"{override.origin}: cannot override engine key {key!r}: the "
                    "standard library module 'std/config' is not loaded",
                    override_program.span,
                )
            )
            continue

        target_index: int | None = None
        target_decl: BuiltinVarDecl | None = None
        for i, item in enumerate(std_config.program.body.items):
            if isinstance(item, BuiltinVarDecl) and item.name == key:
                target_index = i
                target_decl = item
                break
        if target_index is None or target_decl is None:
            diagnostics.append(
                diagnostic_from_span(
                    f"{override.origin}: unknown engine key {key!r}", override_program.span
                )
            )
            continue

        new_items = list(std_config.program.body.items)
        new_items[target_index] = dc_replace(target_decl, default=value_expr)
        new_body = dc_replace(std_config.program.body, items=tuple(new_items))
        new_program = dc_replace(std_config.program, body=new_body)
        std_config = dc_replace(std_config, program=new_program)
        modules[STD_CONFIG_ID] = std_config

    return dc_replace(graph, modules=modules), next_id, diagnostics


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
        return check_program(resolved, capabilities), ()
    except AglError as exc:
        return None, (exc.to_diagnostic(),)
    except Exception as exc:
        diagnostic = Diagnostic(message=f"Type error: {exc}", line=1)
        return None, (diagnostic,)


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
) -> list[tuple["ModuleId", str]]:
    """Return ``(module_id, member_name)`` for every declared extern.

    Scoped externs resolve their unqualified member name in the declaring
    module's companion. The scope pass rejects duplicate scoped symbols, so
    this list remains one-to-one with companion callables.
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
        (mid, funcdef.name)
        for mid, mod in checked.modules.items()
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
    nominals: "Mapping[NominalId, NominalDescriptor] | None" = None,
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
    if nominals is not None:
        registry.set_nominals(dict(nominals))

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
    This runs while reporting an error already in flight, so a field that is
    itself a cyclic array/dict (e.g. a user exception's own data payload), or
    a field of a kind with no JSON representation (``unit``, ``agent``,
    ``constructor``, ``function``, ``iterator`` — legal on an exception field
    even though a cast to ``json`` of such a type is statically rejected),
    must not raise and mask the real error — that one field is reported as a
    marker instead.

    *span* is the optional raise-site source span threaded from ``AglRaise``;
    when present, ``RunError.line`` and ``RunError.col`` are populated from it
    so the CLI can include the source location in its exit-2 error output.
    """
    from agm.agl.ir.ids import Location
    from agm.agl.runtime.serialize import AglNonDataValue, degraded_marker, value_to_json_obj
    from agm.agl.semantics.cycles import AglCyclicValue
    from agm.agl.syntax.spans import SourceSpan

    fields: dict[str, object] = {}
    for k, v in exc.fields.items():
        try:
            fields[k] = value_to_json_obj(v)
        except (AglCyclicValue, AglNonDataValue) as field_exc:
            fields[k] = degraded_marker(field_exc)
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
