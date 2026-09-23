"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source — either from the inline ``-c/--command``
argument or from the source file (exit 1 if unreadable), load the
``[exec]`` configuration, construct a ``PipelineDriver`` with the resolved
settings, call ``runtime.run`` (or a static-only dry run under ``--dry-run``),
print diagnostics to stderr, invoke a selected ``program def`` after linked
initializers when the entry declares one, and exit per the exit-code contract.

Warnings (``result.warnings``) and error diagnostics (``result.diagnostics``)
are two separate channels: warnings are printed to stderr like errors but never
affect the exit code; only error-severity diagnostics yield exit 1.  The
diagnostic severity is included in compiler-style output, e.g.
``path.agl:1:5: warning: message`` or ``1:5: error: message`` for inline
``-c/--command`` source.

Exit-code contract:
    0  success (or a clean ``--dry-run`` static check)
    1  pre-execution failure (unreadable file, static errors, argument validation)
    2  program executed but ended with an uncaught AgL exception

Flag notes:
    - ``--strict-json`` controls JSON-codec strictness: when set, agents must
      return exactly one bare JSON value; the default is lenient recovery
      (fence/prose stripping + trivial repair, then strict schema validation).
      A source-level ``strict_json`` call option overrides this default.
    - Trace logging is OFF by default.  ``--trace`` enables it (auto-named path);
      ``--trace-file PATH`` writes to PATH; ``--no-trace`` disables it.  At most one
      of these three flags may be given (mutually exclusive).  ``[exec] trace =
      true`` in config also enables logging; CLI flags override config.
    - ``--default-agent AGENT`` seeds ``std/config::default-agent`` from host Agent
      syntax or a canonical constructor, taking precedence over the qualified program
      table/``[exec] default-agent``.
    - ``--default-sandbox SANDBOX`` seeds ``std/config::default-sandbox`` from host
      AgentSandbox syntax, taking precedence over the qualified program
      table/``[exec] default-sandbox``.
    - A sole entry-module ``program def`` runs after initializers; when several
      are declared, ``-p``/``--program`` selects one by declaration path. A file
      must declare at least one program; inline ``-c`` statements are wrapped in
      a synthetic ``program def main`` before scope resolution.
    - Every loaded entry and library module receives a ``std/prelude`` glob import
      by default (except ``std/prelude`` itself). An explicit import whose expansion
      includes ``std/prelude`` supplies the prelude contribution instead, so plain
      ``import std/prelude`` leaves prelude names qualified-only. ``--no-stdlib``
      disables the automatic import throughout the loaded program. Ordinary imports are
      qualified by default; tails and ``use`` declarations make names bare.
    - A program reads and writes the engine settings (``strict-json``,
      ``default-agent``, ``default-sandbox``, ``timeout``, ``trace``, ``trace-file``)
      through the ``std/config`` module; a ``std/config::KEY := VALUE`` write takes effect
      from its program point onward and overrides the CLI flag, which overrides
      the config-file layer.  ``--max-call-depth`` remains a host/runtime
      recursion guard.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace.  Evaluation and extern
      companion imports are skipped, so broken companion Python files do not
      fail a dry run.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypedDict, TypeVar, assert_never

from agm.agent.session import create_agl_session_host
from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.builtin_vars import is_engine_builtin_var_key
from agm.agl.modules.roots import RootSet
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.engine_config import restamp_engine_setting
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.option import option_text
from agm.agl.runtime.types import ProgramDeclInfo
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES
from agm.agl.semantics.values import BoolValue, RecordValue
from agm.agl.syntax.nodes import FuncDef, static_items
from agm.cli_support.args import ExecArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds
from agm.cli_support.exec_roots import effective_exec_roots_or_none
from agm.cli_support.exec_target import (
    ExecTargetError,
    PackageProgramReference,
    resolve_installed_reference,
)
from agm.cli_support.param_config import (
    ParamValueTiers,
    _report_undeclared_config_keys,
    resolve_param_values,
)
from agm.cli_support.program_discovery import (
    ProgramDiscoveryArtifacts,
    discover_program_artifacts_for_target,
    program_candidates,
    select_declared_program,
    select_entry_program,
    unmatched_program_message,
)
from agm.cli_support.program_options import (
    EXEC_RESERVED_FLAGS,
    REGISTERED_RESERVED_FLAGS,
    DuplicateOptionFlagError,
    ProgramCommand,
    ProgramHelpRequested,
    ProgramOptionError,
    ReservedFlagError,
    build_program_command,
    native_raw_value,
)
from agm.config.context import ConfigContext, current_config_context
from agm.config.general import exec_config_from_merged, load_general_config
from agm.config.qualified_keys import (
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    resolve_qualified_values,
)
from agm.core import dry_run
from agm.core.cleanup import preserve_primary_error
from agm.core.fs import read_text_arg
from agm.core.log import (
    LiveTracePathResolver,
    prepare_trace_log_from_decision,
    resolve_trace_decision,
)
from agm.core.parse import parse_timeout
from agm.core.toml import toml_dict
from agm.packages.activation import load_activation_index
from agm.packages.manifest import command_paths_for_program
from agm.packages.model import owning_package

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agm.agl.ir.nodes import UseDefault
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.ir.static_keys import StaticBindingKey
    from agm.agl.pipeline import ArgumentPreflight
    from agm.agl.runtime.types import ParamBindingInfo
    from agm.agl.semantics.values import Value
    from agm.config.general import GeneralConfig


class RegisteredProgramUsageError(Exception):
    """Raised when CLI tokens fail to parse against a program's own value parameters.

    Wraps a ``ProgramCommand.parse`` failure. Carries the
    parse-failure message and the selected ``program def``'s own declaration
    (when one was selected) so the caller can render a usage message
    appropriate to how the program was invoked — a plain ``agm exec`` usage
    error, or (for a dispatched registered command) the shared
    registered-command help rendering — without a second discovery pass over
    the same source.
    """

    def __init__(
        self,
        message: str,
        program: ProgramDeclInfo | None = None,
        command: ProgramCommand | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.program = program
        self.command = command


class _PreparedRunOverrides(TypedDict, total=False):
    """Optional typed seed channel passed to the prepared execution."""

    param_seeds: "Mapping[StaticBindingKey, Value]"


def _bind_host_inputs(
    *,
    tokens: "Sequence[str]",
    program: ProgramDeclInfo | None,
    params: "Sequence[ParamBindingInfo]",
    reserved_flags: frozenset[str],
    config: "GeneralConfig",
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
) -> tuple[ProgramArguments, ParamValueTiers]:
    """Bind one selected program's CLI, environment, and config host inputs.

    This is the only execution-host path that combines the signature and
    module-parameter surfaces.  Registered commands reach it through
    :func:`run`, retaining their smaller reserved-flag inventory while sharing
    parsing, config precedence, warning reports, and preflight input shape.

    The returned :class:`ParamValueTiers` is handed to
    ``PipelineDriver.preflight_arguments`` unmerged: it, not this function,
    ranks the module-route (``lower``) tier beneath the selected program's own
    ``@config`` values, which are only known once the program is lowered.
    """
    if program is None:
        if tokens:
            raise RegisteredProgramUsageError(f"unexpected argument: {tokens[0]!r}")
        return ProgramArguments(positional=(), named={}), ParamValueTiers(upper={}, lower={})

    command_result = build_program_command(program, reserved_flags, params)
    if not isinstance(command_result, ProgramCommand):
        print(f"Error: {_program_option_error_message(command_result)}", file=sys.stderr)
        raise SystemExit(1)
    program_command = command_result
    try:
        parsed_tail = program_command.parse(tokens)
    except ProgramHelpRequested as exc:
        raise ProgramHelpRequested(exc.command, program) from exc
    except ValueError as exc:
        raise RegisteredProgramUsageError(str(exc), program, program_command) from exc

    program_named = dict(parsed_tail.arguments.named)
    program_path = (*program.scope_path, program.name)
    argument_options = tuple(
        (
            info,
            projected,
            QualifiedConfigKey(entry_segments, program_path, info.cli.name, command_paths),
        )
        for info, projected in program_command.options
    )
    argument_keys = tuple(key for _info, _projected, key in argument_options)
    try:
        param_tiers, route_reports = resolve_param_values(
            config,
            program,
            parsed_tail.params,
            entry_segments=entry_segments,
            command_paths=command_paths,
            surface=program_command.surface,
        )
    except QualifiedConfigLookupError as exc:
        print(f"Error: invalid qualified configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if entry_segments:
        try:
            configured_arguments = resolve_qualified_values(config, argument_keys)
        except QualifiedConfigLookupError as exc:
            print(f"Error: invalid qualified configuration: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        positional_names = program_command.positionally_filled_names(
            len(parsed_tail.arguments.positional)
        )
        cli_supplied_names = set(program_named) | positional_names
        for info, projected, key in argument_options:
            if key in configured_arguments and info.name not in cli_supplied_names:
                program_named[info.name] = native_raw_value(projected, configured_arguments[key])

    _report_undeclared_config_keys(config, route_reports)
    return ProgramArguments(
        positional=parsed_tail.arguments.positional, named=program_named
    ), param_tiers


def _package_entry_segments(entry_path: Path | None, roots: RootSet) -> tuple[str, ...] | None:
    """Return a package entry's package-qualified module path.

    *roots* carries the mounted package selection it was assembled from, so a
    development checkout and an installed store tree route their configuration
    identically — and identically to the same program reached by its
    ``PACKAGE/MODULE::PROGRAM`` reference. The standard library is included: a
    directly executed ``std`` module routes under ``std/MODULE`` from whichever
    tree was selected as the standard library. This is the module identity the
    loader keys such an entry by, asked of the same root set.
    """
    if entry_path is None:
        return None
    module_id = roots.package_module_id_for(entry_path)
    return None if module_id is None else module_id.segments


def _registered_command_paths(
    entry_path: Path,
    roots: RootSet,
    module_segments: tuple[str, ...],
    program_path: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Return the CLI command paths the entry's own package registers for this program.

    A registered command path addresses the program it names, so it is one
    more spelling of that program's configuration table — read whether the
    program was reached as the command, by installed reference, or by file
    path. The owning package's manifest is the authority dispatch itself
    checks, so a program no package owns has no command table.
    """
    package = owning_package(entry_path, roots.packages)
    if package is None:
        return ()
    reference = "::".join(("/".join(module_segments), *program_path))
    return command_paths_for_program(package.manifest, reference)


_T = TypeVar("_T")


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)


def _option_text(value: "Value | None") -> str | None:
    """Read a standard-identity ``Option[text]`` engine value's payload, if present."""
    if not isinstance(value, RecordValue):
        return None
    return option_text(value, nominals=NO_BUILTIN_DECLARATIONS)


def _registered_command_mismatch(command_path: str) -> NoReturn:
    """Report a cached command that does not agree with its active package."""
    print(
        f"Error: registered command {command_path!r} does not match its active package.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _program_option_error_message(error: ProgramOptionError) -> str:
    """Render one program-parameter CLI-flag collision as a host diagnostic message."""
    match error:
        case ReservedFlagError(parameter=parameter, flag=flag):
            return f"program parameter {parameter!r} projects onto the reserved option {flag!r}"
        case DuplicateOptionFlagError(first_parameter=first, second_parameter=second, flag=flag):
            return f"program parameters {first!r} and {second!r} both project onto option {flag!r}"
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _resolve_registered_program_target(
    program: str, package_name: str, *, context: ConfigContext
) -> PackageProgramReference | None:
    """Resolve a registered program's installed reference, verifying its owning package.

    Returns ``None`` (rather than raising) on any resolution failure or a
    package mismatch, so program-declaration discovery below can degrade to
    an empty inventory instead of surfacing a resolution error from an
    advisory help/completion path.
    """
    target = resolve_installed_reference(
        program, home=context.home, proj_dir=context.proj_dir, cwd=context.cwd
    )
    if not isinstance(target, PackageProgramReference):
        return None
    if target.module_id.segments[0] != package_name:
        return None
    return target


def registered_program_declaration(
    program: str,
    package_name: str,
    *,
    context: ConfigContext | None = None,
    artifact_sink: list[ProgramDiscoveryArtifacts] | None = None,
) -> ProgramDeclInfo | None:
    """Discover the referenced program's own ``program def`` declaration, degrading on failure.

    A registered command names exactly one ``program def`` declaration — the
    one selected here by matching the installed reference's own declaration
    path among the entry module's own candidates, via
    :func:`~agm.cli_support.program_discovery.select_entry_program`, which
    matches a requested name against entry-module declarations. An imported
    module's same-named declaration never shadows it.
    """
    try:
        if context is None:
            context = current_config_context()
        target = _resolve_registered_program_target(program, package_name, context=context)
        if target is None:
            return None
        artifacts = discover_program_artifacts_for_target(
            file=program,
            command=None,
            module_paths=None,
            no_stdlib=False,
            context=context,
        )
        if artifacts is None or artifacts.entry_path != target.entry_path:
            return None
        if artifact_sink is not None:
            artifact_sink.append(artifacts)
        return select_entry_program(
            artifacts.discovery.programs, requested=target.declaration_path
        ).selected
    except (Exception, SystemExit):
        return None


def run(
    args: ExecArgs,
    *,
    entry_module_segments: tuple[str, ...] | None = None,
    reserved_flags: frozenset[str] = EXEC_RESERVED_FLAGS,
) -> None:
    """Run an AgL program selected by an exec argument container.

    *reserved_flags* is the invoking surface's flag inventory, which the
    selected program's parameters may not claim.
    """
    # The program source comes either from an inline ``-c/--command`` argument
    # or from a file.  The CLI layer guarantees exactly one is provided; the
    # defensive ``else`` keeps ``run`` safe when called directly.
    if args.command is not None:
        source = args.command
        entry_path: Path | None = None
        diagnostic_source_name: str | None = None
    elif args.file is not None:
        source = read_text_arg(Path(args.file))
        entry_path = Path(args.file)
        diagnostic_source_name = args.file
    else:
        print("Error: exec requires either a FILE or -c/--command", file=sys.stderr)
        raise SystemExit(1)

    ctx = current_config_context()
    # Module roots are assembled ONCE here, up front: this same root set is
    # reused below for scoping the graph, and its mounted packages classify a
    # directly executed package file so it keeps its package-qualified config
    # route.
    exec_roots = effective_exec_roots_or_none(
        entry_path=entry_path,
        module_paths=args.module_paths,
        cwd=ctx.cwd,
        home=ctx.home,
        proj_dir=ctx.proj_dir,
    )
    if exec_roots is None:
        raise SystemExit(1)

    entry_stem: str | None = Path(args.file).stem if args.file is not None else None
    package_entry_segments = _package_entry_segments(entry_path, exec_roots.roots)
    config_entry_segments: tuple[str, ...]
    if entry_module_segments is not None:
        config_entry_segments = entry_module_segments
    elif package_entry_segments is not None:
        config_entry_segments = package_entry_segments
    else:
        config_entry_segments = () if entry_stem is None else (entry_stem,)
    config_view = load_general_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    merged_config = config_view.merged

    # Inline source remains a statement-oriented host. Its entry parse wraps the
    # AST before scope resolution whenever it has no explicit program entry.
    cached_pipeline = (
        args.pipeline_cache
        if isinstance(args.pipeline_cache, ProgramDiscoveryArtifacts)
        and args.pipeline_cache.source == source
        and args.pipeline_cache.entry_path == entry_path
        and args.pipeline_cache.roots == exec_roots.roots
        and args.pipeline_cache.default_stdlib == (not args.no_stdlib)
        else None
    )
    parsed = (
        cached_pipeline.parsed
        if cached_pipeline is not None
        else PipelineDriver.parse_entry(
            source, entry_path=entry_path, inline_command=args.command is not None
        )
    )

    # Loose file entries address declarations under the file stem; package
    # entries retain their package-qualified route.
    parsed_items = (
        tuple(static_items(parsed.program.body.items)) if parsed.program is not None else ()
    )
    parsed_programs = tuple(
        item for item in parsed_items if isinstance(item, FuncDef) and item.is_program
    )
    # The engine settings a program's own config table overrides must be known
    # before the pipeline that discovers declarations can be built, so this
    # pre-pass applies ``select_declared_program``'s rule to the parsed AST.
    # ``select_entry_program`` still makes the authoritative selection below.
    selected_parsed_program = select_declared_program(
        parsed_programs,
        requested=args.program,
        declaration_path=lambda item: "::".join(
            (*(segment.name for segment in item.scope_path), item.name)
        ),
    )
    engine_program_table: dict[str, object] = {}
    command_paths: tuple[tuple[str, ...], ...] = ()
    if entry_path is not None and selected_parsed_program is not None:
        program_path = tuple(segment.name for segment in selected_parsed_program.scope_path) + (
            selected_parsed_program.name,
        )
        command_paths = _registered_command_paths(
            entry_path, exec_roots.roots, config_entry_segments, program_path
        )
        engine_keys = tuple(
            QualifiedConfigKey(config_entry_segments, program_path, key, command_paths)
            for key in ENGINE_KEY_NAMES
        )
        try:
            engine_program_table = {
                key.leaf: value
                for key, value in resolve_qualified_values(config_view, engine_keys).items()
            }
        except QualifiedConfigLookupError as exc:
            print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    try:
        config = exec_config_from_merged(merged_config, program_table=engine_program_table)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Resolve max call depth: CLI > config.  ``None`` (nothing set at
    # any layer) lets the driver apply its canonical default.
    resolved_call_depth_limit = _first(
        args.max_call_depth,
        config.max_call_depth,
    )

    # Seed only settings explicitly controlled by CLI/config. Runtime fallbacks
    # are not seeds: passing them here would suppress a declared
    # ``builtin var`` initializer.  The shared decoder preserves explicit
    # ``None`` values for Option settings such as --no-timeout.  A host Agent
    # value (``--default-agent``/``[exec] default-agent``) decodes through
    # the same shared path as every other key; a bad value exits 1 here,
    # before the module graph is loaded.
    cli_values: dict[str, object | None] = {}
    if args.strict_json is not None:
        cli_values["strict-json"] = args.strict_json
    if args.timeout is not None:
        cli_values["timeout"] = args.timeout
    elif args.no_timeout:
        cli_values["timeout"] = None
    if args.no_trace:
        cli_values["trace"] = False
    elif args.trace:
        cli_values["trace"] = True
    if args.trace_file is not None:
        cli_values["trace-file"] = args.trace_file
    elif args.no_trace_file:
        cli_values["trace-file"] = None
    if args.default_agent is not None:
        cli_values["default-agent"] = args.default_agent
    if args.default_sandbox is not None:
        cli_values["default-sandbox"] = args.default_sandbox

    # strict-json/timeout/trace are resolved only after preflight, below, once a
    # selected program's own ``@config`` entries (ranked between the config
    # tables and the CLI, see ``EngineSeedTiers.merged``) are known. Building
    # these three tiers (never merging them) is possible now: neither
    # ``build_host_engine_seeds`` nor discovery/preflight reads
    # strict-json/timeout/trace's resolved host values.
    process_environment = dict(os.environ)
    engine_tiers = build_host_engine_seeds(
        config=config,
        primary_table=engine_program_table,
        fallback_table=toml_dict(merged_config.get("exec")),
        cli_values=cli_values,
    )

    # Load + scope the graph ONCE, against the module roots assembled above. A
    # source ``std/config::KEY := VALUE`` write takes effect at its program
    # point and overrides the CLI flag, which overrides the config-file layer.
    prepared = (
        cached_pipeline.prepared
        if cached_pipeline is not None
        else PipelineDriver.prepare_parsed_entry(
            parsed,
            roots=exec_roots.roots,
            default_stdlib=not args.no_stdlib,
        )
    )

    # ``prepare_parsed_entry`` was already called above; the same ``PreparedProgram``
    # is reused for discovery and the run, so the source is loaded and scoped only once.
    # strict-json/agent-dispatch/session/timeout are wired in below, once the
    # selected program's own ``@config`` entries are known (see
    # ``configure_execution_services``); discovery and preflight never read them.
    runtime = PipelineDriver(default_call_depth_limit=resolved_call_depth_limit)
    discovery = (
        cached_pipeline.discovery
        if cached_pipeline is not None
        and prepared is cached_pipeline.prepared
        and cached_pipeline.discovery.compiled is not None
        and cached_pipeline.discovery.compiled.capabilities
        == runtime.host_environment().capabilities
        else runtime.discover_programs(prepared)
    )
    for diag in discovery.warnings:
        print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
    checked = discovery.checked
    if checked is None or discovery.diagnostics:
        for diag in discovery.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    # ``select_entry_program`` matches a requested ``-p``/``--program`` name
    # against the entry module's own declarations, shared with
    # ``cli._exec_print_help``'s degraded help rendering, so the two surfaces
    # can never disagree about which program a given name selects.
    selection = select_entry_program(discovery.programs, requested=args.program)
    entry_programs = selection.entry_programs
    if args.file is not None and not entry_programs:
        print("Error: file must declare at least one program.", file=sys.stderr)
        raise SystemExit(1)

    selected_program = selection.selected
    if selected_program is None:
        if selection.requested_unmatched:
            print(unmatched_program_message(args.program, entry_programs), file=sys.stderr)
            raise SystemExit(1)
        if len(entry_programs) > 1:
            candidates = program_candidates(entry_programs)
            print(
                f"Error: multiple programs declared; select one with -p: {candidates}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    arguments, param_tiers = _bind_host_inputs(
        tokens=args.argument_tokens,
        program=selected_program,
        params=() if selected_program is None else discovery.params_for(selected_program),
        reserved_flags=reserved_flags,
        config=config_view,
        entry_segments=config_entry_segments,
        command_paths=command_paths,
    )

    # Program arguments are validated against the lowered program, so this
    # preflight lowers the graph.  It must report a failure (exit 1) BEFORE
    # the trace file is prepared and the runner is built — hence a
    # check-only pass here rather than letting the run below surface it.
    # The lowered program it produces is handed to that run, so the graph
    # is lowered exactly once per invocation.
    executable: "ExecutableProgram | None" = None
    program_symbol = None
    arguments_bound: "tuple[Value | UseDefault, ...] | None" = None
    argument_preflight: "ArgumentPreflight | None" = None
    if selected_program is not None:
        argument_preflight = runtime.preflight_arguments(
            prepared,
            selected_program,
            arguments,
            compiled=discovery.compiled,
            param_values=param_tiers.upper,
            param_values_lower=param_tiers.lower,
        )
        if not argument_preflight.result.ok:
            for diag in argument_preflight.result.diagnostics:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            raise SystemExit(1)
        executable = argument_preflight.executable
        assert executable is not None
        program_symbol = executable.program_symbols[selected_program.node_id]
        arguments_bound = argument_preflight.arguments

    # The selected program's own ``@config`` entries (evaluated by preflight
    # above) rank between the config tables and the CLI for both a module
    # parameter (folded into ``param_seeds`` by preflight already, filtered by
    # ``key in executable.param_bindings``) and an engine setting (folded in
    # here, filtered by ``is_engine_builtin_var_key`` — the checker admits only
    # these two target kinds, so every entry lands in exactly one filter).
    # Resolving these only now — after a preflight failure has already exited
    # 1 — is what lets ``@config`` reach them without a second lowering pass
    # or a duplicated precedence rule. A ``@config`` value is decoded against
    # *executable*'s own nominal identity (see ``preflight_arguments``); an
    # enum-backed value (``timeout``, ``trace-file``, ``default-agent``,
    # ``default-sandbox``) is
    # restamped onto the standard identity every other engine tier already
    # uses, so it reads back through the same plain accessors.
    config_engine_values: dict[str, Value] = {}
    if executable is not None:
        assert argument_preflight is not None
        config_engine_values = {
            key[2]: restamp_engine_setting(
                key[2],
                value,
                from_table=executable.builtin_nominals,
                to_table=NO_BUILTIN_DECLARATIONS,
            )
            for key, value in argument_preflight.program_config.items()
            if is_engine_builtin_var_key(key)
        }
    engine_seeds = engine_tiers.merged(middle=config_engine_values)

    strict_seed = engine_seeds.get("strict-json")
    resolved_strict_json = isinstance(strict_seed, BoolValue) and strict_seed.value

    timeout_text = _option_text(engine_seeds.get("timeout"))
    if timeout_text is not None:
        try:
            resolved_timeout: float | None = parse_timeout(timeout_text)
        except ValueError as exc:
            origin = "--timeout" if cli_values.get("timeout") is not None else "@config timeout"
            print(f"Error: invalid {origin} value: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    else:
        resolved_timeout = None

    # Config tables and ``@config`` (never the CLI — see
    # ``EngineSeedTiers.config_merged``) feed the derived ``trace`` rule the
    # same way ``EngineSeedTiers._resolve_trace`` does.
    config_result = engine_tiers.config_merged(config_engine_values)
    config_trace_value = config_result.get("trace")
    config_trace = isinstance(config_trace_value, BoolValue) and config_trace_value.value
    config_trace_file = _option_text(config_result.get("trace-file"))
    trace_decision = resolve_trace_decision(
        cli_no_trace=args.no_trace,
        cli_trace=args.trace,
        cli_trace_file=args.trace_file,
        config_trace=config_trace,
        config_trace_file=config_trace_file,
    )

    factory = value_driven_agent_factory(idle_timeout=resolved_timeout, context=ctx)
    session_host = create_agl_session_host(idle_timeout=resolved_timeout, context=ctx)
    runtime.configure_execution_services(
        default_strict_json=resolved_strict_json,
        agent_dispatcher=factory,
        session_host=session_host,
        shell_exec_timeout=resolved_timeout,
    )

    # Resolve + validate the trace log file up front.  --dry-run is
    # side-effect-free: no trace is written regardless of --trace-file.  A source
    # ``std/config::trace``/``trace-file`` write takes effect at runtime via the
    # host reconfigurer, not here.
    if dry_run.enabled():
        trace_file = None
    else:
        trace_file = prepare_trace_log_from_decision(trace_decision, command_name="exec")

    policy = HostSettingsPolicy(
        resolve_trace_path=LiveTracePathResolver(command_name="exec", auto_path=trace_file),
    )

    # Warnings live on their own channel and never affect the exit code;
    # ``result.diagnostics`` holds only error-severity pre-execution failures.
    # Warnings carry a ``warning:`` prefix to disambiguate them from error
    # diagnostics on the shared stderr channel.
    printed_warnings = {
        (
            diag.line,
            diag.column,
            diag.end_line,
            diag.end_column,
            diag.message,
            diag.severity,
        )
        for diag in discovery.warnings
    }

    # Reuse the ``PreparedProgram`` from above — no second parse/scope of the source.
    # Pass the already-computed compiled from discovery and the program the
    # preflight already lowered, so the graph is type-checked, match-compiled and
    # lowered exactly once. Keep result-to-exit handling inside the cleanup
    # boundary: a failed result is a primary program failure, just like an
    # exception, and must not be replaced by a secondary close failure.
    with preserve_primary_error(session_host.close_all, label="agent session cleanup"):
        run_kwargs: _PreparedRunOverrides = {}
        if argument_preflight is not None and argument_preflight.param_seeds:
            run_kwargs["param_seeds"] = argument_preflight.param_seeds
        result = runtime.run_prepared(
            prepared,
            check_only=dry_run.enabled(),
            trace_file=trace_file,
            compiled=discovery.compiled,
            executable=executable,
            host_settings_policy=policy,
            builtin_host_settings=engine_seeds,
            process_environment=process_environment,
            program_symbol=program_symbol,
            arguments=arguments_bound,
            **run_kwargs,
        )

        for diag in result.warnings:
            warning_key = (
                diag.line,
                diag.column,
                diag.end_line,
                diag.end_column,
                diag.message,
                diag.severity,
            )
            if warning_key not in printed_warnings:
                print(
                    format_diagnostic(diag, source_name=diagnostic_source_name),
                    file=sys.stderr,
                )

        if result.ok:
            # Print the static call-site inventory when running under --dry-run.
            if dry_run.enabled() and result.call_sites:
                print("call-sites:")
                for site in result.call_sites:
                    schema_tag = ", schema: yes" if site.has_schema else ""
                    policy_tag = (
                        f", policy: {site.parse_policy}" if site.parse_policy != "default" else ""
                    )
                    print(
                        f"  line {site.line}:{site.col}: {site.callee} "
                        f"→ {site.target_type} "
                        f"[{site.codec_name}{schema_tag}{policy_tag}]"
                    )
            return

        # Pre-execution failure: print error diagnostics and exit 1.
        if result.error is None:
            for diag in result.diagnostics:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            raise SystemExit(1)

        # Uncaught AgL exception: print and exit 2.
        print(result.error.to_message(), file=sys.stderr)
        raise SystemExit(2)


def _resolve_installed_reference_or_exit(
    program: str, *, context: ConfigContext, package_name: str | None = None
) -> PackageProgramReference:
    """Resolve an installed reference through the shared resolver, exiting on failure."""
    target = resolve_installed_reference(
        program,
        home=context.home,
        proj_dir=context.proj_dir,
        cwd=context.cwd,
        package_name=package_name,
    )
    if isinstance(target, ExecTargetError):
        print(f"Error: {target.message}", file=sys.stderr)
        raise SystemExit(1)
    return target


def run_registered(
    program: str,
    argument_tokens: list[str],
    *,
    args: ExecArgs | None = None,
    package: str | None = None,
    command_path: str | None = None,
    pipeline_cache: ProgramDiscoveryArtifacts | None = None,
) -> None:
    """Run an installed reference, verifying a registered command against its manifest.

    The activation index only locates a candidate command. Registered dispatch
    supplies its cached package and command path, which are checked against the
    active (and project-pinned) package manifest before execution. Direct
    ``agm exec PACKAGE/MODULE::PROGRAM`` references have no command-path
    restriction, but still require that their module is owned by the selected
    active package, and reserve ``agm exec``'s flags rather than a registered
    command's.
    """
    if (package is None) != (command_path is None):
        print("Error: incomplete registered package command metadata.", file=sys.stderr)
        raise SystemExit(1)

    context = current_config_context()
    target = _resolve_installed_reference_or_exit(program, context=context, package_name=package)

    if package is not None and command_path is not None:
        from agm.packages.manifest import expanded_commands

        command = expanded_commands(target.package.manifest).get(command_path)
        if command is None or command.program is None:
            _registered_command_mismatch(command_path)
        if command.program != program:
            # The index deliberately remains an install-time cache. Editable
            # packages are its one live exception: their manifest is reread
            # above with the active package selection, so follow a changed
            # registration while immutable cached entries stay fail-closed.
            active = load_activation_index(home=context.home).packages.get(package)
            if active is None or active.editable != target.package.root:
                _registered_command_mismatch(command_path)
            program = command.program
            target = _resolve_installed_reference_or_exit(
                program, context=context, package_name=package
            )

    if not target.entry_path.is_file():
        print(f"Error: installed program reference {program!r} was not found.", file=sys.stderr)
        raise SystemExit(1)
    if owning_package(target.entry_path, target.packages) is not target.package:
        print(
            f"Error: installed program reference {program!r} is not owned by its active package.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    execution_args = (
        ExecArgs(
            file=program,
            argument_tokens=argument_tokens,
            strict_json=None,
            no_trace=False,
            trace_file=None,
        )
        if args is None
        else args
    )
    run(
        replace(
            execution_args,
            file=str(target.entry_path.resolve()),
            program=(
                target.declaration_path
                if execution_args.program is None
                else execution_args.program
            ),
            pipeline_cache=(
                pipeline_cache if pipeline_cache is not None else execution_args.pipeline_cache
            ),
        ),
        entry_module_segments=target.module_id.segments,
        reserved_flags=(EXEC_RESERVED_FLAGS if command_path is None else REGISTERED_RESERVED_FLAGS),
    )
