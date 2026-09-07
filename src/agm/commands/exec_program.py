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
    - Trace logging is OFF by default.  ``--log`` enables it (auto-named path);
      ``--log-file PATH`` writes to PATH; ``--no-log`` disables it.  At most one
      of these three flags may be given (mutually exclusive).  ``[exec] log =
      true`` in config also enables logging; CLI flags override config.
    - ``--agent AGL_LITERAL`` seeds ``std/config::default-agent`` from one typed
      constant Agent expression, taking precedence over the qualified program
      table/``[exec] default-agent``, which in turn takes precedence over the
      bare host command in ``[exec] runner``.
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
      ``max-iters``, ``default-agent``, ``timeout``, ``log``, ``log-file``) through the
      ``std/config`` module; a ``std/config::KEY := VALUE`` write takes effect
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
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypeVar, assert_never

from agm.agent.session import create_agl_session_host
from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.roots import RootSet
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.arguments import ProgramArguments
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.types import ProgramDeclInfo
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES
from agm.agl.syntax.nodes import FuncDef, static_items
from agm.cli_support.args import ExecArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds, check_max_iters
from agm.cli_support.exec_roots import effective_exec_roots_or_none
from agm.cli_support.exec_target import (
    ExecTargetError,
    PackageProgramReference,
    resolve_installed_reference,
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
    DuplicateOptionFlagError,
    ProgramCommand,
    ProgramHelpRequested,
    ProgramOptionError,
    ReservedFlagError,
    build_program_command,
    native_raw_value,
)
from agm.config.context import ConfigContext, current_config_context
from agm.config.general import GeneralConfig, exec_config_from_merged, load_general_config
from agm.config.qualified_keys import (
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    configured_leaf_tables,
    display_table_path,
    resolve_qualified_values,
)
from agm.core import dry_run
from agm.core.cleanup import preserve_primary_error
from agm.core.fs import read_text_arg
from agm.core.log import (
    LiveTracePathResolver,
    prepare_trace_log_from_decision,
    resolve_log_decision,
)
from agm.core.parse import parse_timeout
from agm.core.toml import toml_dict
from agm.packages.activation import load_activation_index
from agm.packages.manifest import command_paths_for_program
from agm.packages.model import owning_package

if TYPE_CHECKING:
    from agm.agl.ir.nodes import UseDefault
    from agm.agl.ir.program import ExecutableProgram
    from agm.agl.semantics.values import Value


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
    ) -> None:
        super().__init__(message)
        self.message = message
        self.program = program


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


def _registered_command_mismatch(command_path: str) -> NoReturn:
    """Report a cached command that does not agree with its active package."""
    print(
        f"Error: registered command {command_path!r} does not match its active package.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _report_undeclared_config_keys(
    config: GeneralConfig,
    module_segments: tuple[str, ...],
    argument_keys: Iterable[QualifiedConfigKey],
    *,
    scope_path: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    positional_only_names: Iterable[str],
) -> None:
    """Warn about config keys in a program's qualified table that no argument claims.

    A key set in the table that is neither one of the selected program's own
    value parameters nor an engine setting is read by nothing — most often a
    misspelled name — so report it instead of dropping it silently. The
    warning never affects the run: the program still executes on its
    defaults. Engine settings legitimately share the table with program
    arguments, and nested tables (scope regions, per-program engine settings)
    address routes of their own.

    *scope_path* is the selected program's own qualified table path (e.g.
    ``workflow.main``); *command_paths* adds the CLI paths the program is
    registered under, which address it as well. Each warning names the table
    the key was actually read from, so it points at the spelling its author
    wrote.

    *positional_only_names* names leaves that are declared but not
    name-addressable — a positional-only program argument, which a config
    table (a name-keyed channel) can never supply, the same way it can never
    be passed by name in an AgL call. Such a leaf is a distinct, milder
    warning than a genuinely undeclared key: it exists, it is just not
    reachable by this channel.
    """
    declared = {key.leaf for key in argument_keys}
    positional_only = set(positional_only_names)
    leaf_tables = configured_leaf_tables(config, module_segments, scope_path, command_paths)
    for leaf in sorted(leaf_tables):
        if leaf in declared or leaf in ENGINE_KEY_NAMES:
            continue
        table_name = display_table_path(leaf_tables[leaf])
        if leaf in positional_only:
            print(
                f"warning: config key '{leaf}' in the '{table_name}' configuration table "
                "names a positional-only program argument, which can only be supplied "
                "positionally, and will be ignored",
                file=sys.stderr,
            )
            continue
        print(
            f"warning: config key '{leaf}' in the '{table_name}' "
            "configuration table is not a declared program argument and will be ignored",
            file=sys.stderr,
        )


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
) -> None:
    """Run an AgL program selected by an exec argument container."""
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

    # Inline source remains a statement-oriented host. Its AST is wrapped before
    # scope resolution whenever it has no explicit program entry.
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
        else PipelineDriver.parse_entry(source, entry_path=entry_path)
    )
    if cached_pipeline is None and args.command is not None and parsed.program is not None:
        from agm.agl.parser import wrap_inline_program

        program, next_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)
        parsed = replace(parsed, program=program, next_id=next_id)

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

    # Resolve strict_json: CLI > config. A source ``std/config::strict-json :=
    # VALUE`` write is applied at runtime when it updates the live setting.
    strict_json = _first(args.strict_json, config.strict_json)
    # config.strict_json is always a bool, so _first always returns a bool here.
    assert strict_json is not None
    resolved_strict_json: bool = strict_json

    # Resolve max call depth: CLI > config.  ``None`` (nothing set at
    # any layer) lets the driver apply its canonical default.
    resolved_call_depth_limit = _first(
        args.max_call_depth,
        config.max_call_depth,
    )

    # Resolve loop limit (max-iters valve): CLI > config. ``None`` leaves the
    # valve off. A source ``std/config::max-iters := VALUE`` write is applied
    # at runtime from its program point, overriding this initial value.
    check_max_iters(args.max_iters)
    resolved_loop_limit = _first(args.max_iters, config.default_loop_limit)

    # Resolve timeout: CLI > [exec] config. A source ``std/config::timeout :=
    # VALUE`` write is applied at runtime from its program point.
    # ``--timeout VALUE`` overrides the config; ``--no-timeout`` clears it (None).
    if args.timeout is not None:
        try:
            resolved_timeout: float | None = parse_timeout(args.timeout)
        except ValueError as exc:
            print(f"Error: invalid --timeout value: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    elif args.no_timeout:
        resolved_timeout = None
    else:
        resolved_timeout = config.timeout

    factory = value_driven_agent_factory(idle_timeout=resolved_timeout)
    session_host = create_agl_session_host(idle_timeout=resolved_timeout)

    # Resolve the CLI > config logging decision ONCE: it both drives the trace
    # file prepared below and seeds the readable ``log`` register.
    log_decision = resolve_log_decision(
        cli_no_log=args.no_log,
        cli_log=args.log,
        cli_log_file=args.log_file,
        config_log=config.log,
        config_log_file=config.log_file,
    )

    # Seed only settings explicitly controlled by CLI/config. Runtime fallbacks
    # are not seeds: passing them here would suppress a declared
    # ``builtin var`` initializer.  The shared decoder preserves explicit
    # ``None`` values for Option settings such as --no-timeout.  An AgL agent
    # literal (``--agent``/``[exec] default-agent``) becomes an override
    # spliced into the program's own compilation below rather than a seed
    # value; a bad literal exits 1 here, before the module graph is loaded.
    cli_values: dict[str, object | None] = {}
    if args.strict_json is not None:
        cli_values["strict-json"] = args.strict_json
    if args.max_iters is not None:
        cli_values["max-iters"] = args.max_iters
    if args.timeout is not None:
        cli_values["timeout"] = args.timeout
    elif args.no_timeout:
        cli_values["timeout"] = None
    if args.no_log:
        cli_values["log"] = False
    elif args.log:
        cli_values["log"] = True
    if args.log_file is not None:
        cli_values["log-file"] = args.log_file
    elif args.no_log_file:
        cli_values["log-file"] = None

    process_environment = dict(os.environ)
    engine_seeds = build_host_engine_seeds(
        config=config,
        primary_table=engine_program_table,
        fallback_table=toml_dict(merged_config.get("exec")),
        cli_values=cli_values,
        agent=args.agent,
    )

    # Load + scope the graph ONCE, against the module roots assembled above,
    # splicing any engine-setting overrides in as part of that same pass.  A
    # source ``std/config::KEY := VALUE`` write takes effect at its program
    # point and overrides the CLI flag, which overrides the config-file layer.
    prepared = (
        cached_pipeline.prepared
        if cached_pipeline is not None and not engine_seeds.overrides
        else PipelineDriver.prepare_parsed_entry(
            parsed,
            roots=exec_roots.roots,
            default_stdlib=not args.no_stdlib,
            setting_overrides=engine_seeds.overrides,
        )
    )

    # ``prepare_parsed_entry`` was already called above; the same ``PreparedProgram``
    # is reused for discovery and the run, so the source is loaded and scoped only once.
    runtime = PipelineDriver(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        agent_dispatcher=factory,
        session_host=session_host,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
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

    # A selected program's own value parameters project onto their own Click
    # command, built from its declared signature. This is a static,
    # source-derived check independent of any supplied arguments, so a
    # colliding projection (against a reserved host flag, or against another
    # parameter's own flag) is reported unconditionally.
    program_command: ProgramCommand | None = None
    if selected_program is not None:
        command_result = build_program_command(selected_program)
        if isinstance(command_result, ProgramCommand):
            program_command = command_result
        else:
            print(f"Error: {_program_option_error_message(command_result)}", file=sys.stderr)
            raise SystemExit(1)

    if program_command is not None:
        try:
            cli_arguments = program_command.parse(args.argument_tokens)
        except ProgramHelpRequested as exc:
            # The command owns ``-h``/``--help``; the caller renders the help
            # its own invocation calls for, so it is given the declaration
            # behind the command as well.
            raise ProgramHelpRequested(exc.command, selected_program) from exc
        except ValueError as exc:
            raise RegisteredProgramUsageError(str(exc), selected_program) from exc
    elif args.argument_tokens:
        raise RegisteredProgramUsageError(
            f"unexpected argument: {args.argument_tokens[0]!r}", selected_program
        )
    else:
        cli_arguments = ProgramArguments(positional=(), named={})

    # The selected program's own value parameters resolve config-file values
    # from its qualified table (e.g. ``[workflow.main]``), keyed by their
    # external names — the same table an engine-key override reads, and the
    # same ``QualifiedConfigKey`` shape. Precedence is CLI token > ``@opt-env``
    # variable > config table > signature default: a parameter Click filled
    # from either of the first two is already in ``cli_arguments.named``, so
    # config values are folded in only beneath those.
    program_named: dict[str, object] = dict(cli_arguments.named)
    if entry_path is not None and program_command is not None and selected_program is not None:
        program_path = selected_program.scope_path + (selected_program.name,)
        argument_options = tuple(
            (
                info,
                projected,
                QualifiedConfigKey(
                    config_entry_segments, program_path, info.cli.name, command_paths
                ),
            )
            for info, projected in program_command.options
        )
        argument_keys = [key for _info, _projected, key in argument_options]
        try:
            configured_arguments = resolve_qualified_values(config_view, argument_keys)
        except QualifiedConfigLookupError as exc:
            print(f"Error: invalid qualified configuration: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        positional_names = program_command.positionally_filled_names(len(cli_arguments.positional))
        cli_supplied_names = set(program_named) | positional_names
        for info, projected, key in argument_options:
            if key in configured_arguments and info.name not in cli_supplied_names:
                program_named[info.name] = native_raw_value(projected, configured_arguments[key])
        _report_undeclared_config_keys(
            config_view,
            config_entry_segments,
            argument_keys,
            scope_path=program_path,
            command_paths=command_paths,
            positional_only_names=program_command.positional_only_names(),
        )
    arguments = ProgramArguments(positional=cli_arguments.positional, named=program_named)

    # Program arguments are validated against the lowered program, so this
    # preflight lowers the graph.  It must report a failure (exit 1) BEFORE
    # the trace file is prepared and the runner is built — hence a
    # check-only pass here rather than letting the run below surface it.
    # The lowered program it produces is handed to that run, so the graph
    # is lowered exactly once per invocation.
    executable: "ExecutableProgram | None" = None
    program_symbol = None
    arguments_bound: "tuple[Value | UseDefault, ...] | None" = None
    if selected_program is not None:
        argument_preflight = runtime.preflight_arguments(
            prepared, selected_program, arguments, compiled=discovery.compiled
        )
        if not argument_preflight.result.ok:
            for diag in argument_preflight.result.diagnostics:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            raise SystemExit(1)
        executable = argument_preflight.executable
        assert executable is not None
        program_symbol = executable.program_symbols[selected_program.node_id]
        arguments_bound = argument_preflight.arguments

    # Resolve + validate the trace log file up front.  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file.  A source
    # ``std/config::log``/``log-file`` write takes effect at runtime via the host
    # reconfigurer, not here.
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_decision(log_decision, command_name="exec")

    policy = HostSettingsPolicy(
        resolve_trace_path=LiveTracePathResolver(command_name="exec", auto_path=log_file),
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
        result = runtime.run_prepared(
            prepared,
            check_only=dry_run.enabled(),
            log_file=log_file,
            compiled=discovery.compiled,
            executable=executable,
            host_settings_policy=policy,
            builtin_host_settings=engine_seeds.values,
            process_environment=process_environment,
            program_symbol=program_symbol,
            arguments=arguments_bound,
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
    active package.
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
            no_log=False,
            log_file=None,
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
    )
