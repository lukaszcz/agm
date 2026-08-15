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
    1  pre-execution failure (unreadable file, static errors, param validation)
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
    - Every loaded entry and library module opens ``std/core`` by default
      (except ``std/core`` itself). ``--no-stdlib`` disables that automatic
      opening throughout the loaded program. Ordinary imports are qualified by
      default; ``open import`` and ``using`` make selected names bare.
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

import sys
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, TypeVar

from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.types import ParamDeclInfo
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES
from agm.agl.syntax.nodes import FuncDef, ParamDecl, static_items
from agm.cli_support.args import ExecArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds, check_max_iters
from agm.cli_support.exec_params import (
    discover_params_from_installed_reference,
    external_param_keys,
    param_option_flags,
    parse_param_tokens,
)
from agm.cli_support.exec_roots import effective_exec_roots
from agm.cli_support.exec_target import (
    ExecTargetError,
    PackageProgramReference,
    resolve_installed_reference,
)
from agm.config.context import ConfigContext, current_config_context
from agm.config.general import GeneralConfig, exec_config_from_merged, load_general_config
from agm.config.module_roots import StdlibResolutionError
from agm.config.qualified_keys import (
    RESERVED_CONFIG_SECTION_NAMES,
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    build_qualified_config_key,
    configured_leaf_names,
    resolve_qualified_values,
    route_table_paths,
)
from agm.core import dry_run
from agm.core.fs import read_text_arg
from agm.core.log import (
    LiveTracePathResolver,
    prepare_trace_log_from_decision,
    resolve_log_decision,
)
from agm.core.parse import parse_timeout
from agm.core.toml import toml_dict
from agm.packages.activation import load_activation_index
from agm.packages.model import PackageInfo, owning_package


class RegisteredParamUsageError(Exception):
    """Raised when CLI parameter tokens fail to parse against a program's params.

    Carries the parse-failure message and the program's parameter inventory so
    the caller can render a usage message appropriate to how the program was
    invoked: a plain ``agm exec`` usage error, or (for a dispatched registered
    command) the shared registered-command help rendering.
    """

    def __init__(self, message: str, params: tuple[ParamDeclInfo, ...]) -> None:
        super().__init__(message)
        self.message = message
        self.params = params


def _entry_module_segments(
    entry_module_segments: tuple[str, ...], module_segments: tuple[str, ...]
) -> tuple[str, ...]:
    """Replace the entry sentinel with its config module component."""
    if module_segments == ("<entry>",):
        return entry_module_segments
    return module_segments


def _development_entry_segments(
    entry_path: Path | None, packages: tuple[PackageInfo, ...]
) -> tuple[str, ...] | None:
    """Return a development-package entry's package-qualified module path."""
    if entry_path is None:
        return None
    package = owning_package(entry_path, packages)
    if package is None:
        return None
    relative = entry_path.resolve().relative_to(package.root).with_suffix("")
    return relative.parts


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
    param_keys: Iterable[QualifiedConfigKey],
) -> None:
    """Warn about entry-module config keys that no declaration claims.

    A key set in the entry module's configuration table that is neither one of
    its params nor an engine setting is read by nothing — most often a
    misspelled param name — so report it instead of dropping it silently. The
    warning never affects the run: the program still executes on its defaults.
    Engine settings legitimately share the table with params, and nested tables
    (scope regions, per-program engine settings) address routes of their own.

    A param claims a leaf whenever its own route reads the entry module's
    table, which by suffix resolution includes params declared in imported
    modules whose route ends in the entry module's name.
    """
    entry_paths = set(route_table_paths(module_segments))
    declared = {
        key.leaf
        for key in param_keys
        if entry_paths.intersection(route_table_paths(key.module_segments, key.scope_path))
    }
    for leaf in sorted(configured_leaf_names(config, module_segments)):
        if leaf in declared or leaf in ENGINE_KEY_NAMES:
            continue
        print(
            f"warning: config key '{leaf}' in the '{'/'.join(module_segments)}' "
            "configuration table is not a declared param and will be ignored",
            file=sys.stderr,
        )


def registered_program_params(
    program: str, package_name: str, *, context: ConfigContext | None = None
) -> tuple[ParamDeclInfo, ...]:
    """Discover the parameter inventory for a registered program, degrading on failure."""
    try:
        if context is None:
            context = current_config_context()
        target = resolve_installed_reference(
            program, home=context.home, proj_dir=context.proj_dir, cwd=context.cwd
        )
        if not isinstance(target, PackageProgramReference):
            return ()
        if target.module_id.segments[0] != package_name:
            return ()
        return discover_params_from_installed_reference(
            target,
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
        )
    except (Exception, SystemExit):
        return ()


def registered_program_param_flags(
    program: str, package_name: str, *, context: ConfigContext | None = None
) -> tuple[str, ...]:
    """Discover parameter flags for a registered program, degrading on failure."""
    return param_option_flags(registered_program_params(program, package_name, context=context))


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
    # Module roots (and the development packages reachable from the entry) are
    # assembled ONCE here, up front: this same root set is reused below for
    # scoping the graph, and the development packages for routing a directly
    # executed development-package file to its package-qualified config route.
    try:
        exec_roots = effective_exec_roots(
            entry_path=entry_path,
            module_paths=args.module_paths,
            cwd=ctx.cwd,
            home=ctx.home,
            proj_dir=ctx.proj_dir,
        )
    except (StdlibResolutionError, ValueError) as exc:
        print(f"Error: invalid module roots configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    entry_stem: str | None = Path(args.file).stem if args.file is not None else None
    development_entry_segments = _development_entry_segments(
        entry_path, exec_roots.development_packages
    )
    config_entry_segments: tuple[str, ...]
    if entry_module_segments is not None:
        config_entry_segments = entry_module_segments
    elif development_entry_segments is not None:
        config_entry_segments = development_entry_segments
    else:
        config_entry_segments = () if entry_stem is None else (entry_stem,)
    config_view = load_general_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    merged_config = config_view.merged

    # Inline source remains a statement-oriented host. Its AST is wrapped before
    # scope resolution whenever it has no explicit program entry.
    parsed = PipelineDriver.parse_entry(source, entry_path=entry_path)
    if args.command is not None and parsed.program is not None:
        from agm.agl.parser import wrap_inline_program

        program, next_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)
        parsed = replace(parsed, program=program, next_id=next_id)

    # Loose file entries address declarations under the file stem; package
    # entries retain their package-qualified route. A loose stem that would
    # consume AGM's configuration namespace is harmless until it exposes params.
    # Materialized: the items are scanned twice below, for params and programs.
    parsed_items = (
        tuple(static_items(parsed.program.body.items)) if parsed.program is not None else ()
    )
    if (
        len(config_entry_segments) == 1
        and entry_stem in RESERVED_CONFIG_SECTION_NAMES
        and any(isinstance(item, ParamDecl) for item in parsed_items)
    ):
        print(
            f"Error: entry file stem '{entry_stem}' is reserved for configuration.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    parsed_programs = tuple(
        item for item in parsed_items if isinstance(item, FuncDef) and item.is_program
    )
    selected_parsed_program = (
        parsed_programs[0]
        if len(parsed_programs) == 1
        else next(
            (
                item
                for item in parsed_programs
                if "::".join((*(segment.name for segment in item.scope_path), item.name))
                == args.program
            ),
            None,
        )
        if args.program is not None
        else None
    )
    engine_program_table: dict[str, object] = {}
    if entry_stem is not None and selected_parsed_program is not None:
        program_path = tuple(segment.name for segment in selected_parsed_program.scope_path) + (
            selected_parsed_program.name,
        )
        engine_keys = tuple(
            QualifiedConfigKey(config_entry_segments, program_path, key) for key in ENGINE_KEY_NAMES
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
    prepared = PipelineDriver.prepare_parsed_entry(
        parsed,
        roots=exec_roots.roots,
        default_stdlib=not args.no_stdlib,
        setting_overrides=engine_seeds.overrides,
    )

    # ``prepare_parsed_entry`` was already called above; the same ``PreparedProgram``
    # is reused for discovery and the run, so the source is loaded and scoped only once.
    runtime = PipelineDriver(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        agent_dispatcher=factory,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
    discovery = runtime.discover_params(prepared)
    for diag in discovery.warnings:
        print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
    checked = discovery.checked
    if checked is None or discovery.diagnostics:
        for diag in discovery.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    entry_programs = tuple(program for program in discovery.programs if program.module.is_entry)
    if args.file is not None and not entry_programs:
        print("Error: file must declare at least one program.", file=sys.stderr)
        raise SystemExit(1)

    selected_program = None
    if len(entry_programs) == 1:
        selected_program = entry_programs[0]
    elif len(entry_programs) > 1 and args.program is None:
        candidates = ", ".join(program.declaration_path for program in entry_programs)
        print(
            f"Error: multiple programs declared; select one with -p: {candidates}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.program is not None:
        selected_program = next(
            (program for program in entry_programs if program.declaration_path == args.program),
            None,
        )
        if selected_program is None:
            candidates = ", ".join(program.declaration_path for program in entry_programs)
            suffix = f" Candidates: {candidates}" if candidates else ""
            print(f"Error: no program matches '{args.program}'.{suffix}", file=sys.stderr)
            raise SystemExit(1)

    selected_params = (
        discovery.params if selected_program is None else discovery.params_for(selected_program)
    )
    external_params: dict[str, object] = {}
    try:
        cli_params = parse_param_tokens(selected_params, args.param_tokens)
    except ValueError as exc:
        raise RegisteredParamUsageError(str(exc), selected_params) from exc

    if entry_stem is not None:
        param_keys = {
            param: build_qualified_config_key(
                _entry_module_segments(config_entry_segments, param.module_segments), param.name
            )
            for param in selected_params
        }
        try:
            configured_params = resolve_qualified_values(config_view, param_keys.values())
        except QualifiedConfigLookupError as exc:
            print(f"Error: invalid qualified configuration: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        external_keys = external_param_keys(selected_params)
        external_params.update(
            {
                external_keys[param]: configured_params[key]
                for param, key in param_keys.items()
                if key in configured_params
            }
        )
        _report_undeclared_config_keys(config_view, config_entry_segments, param_keys.values())
    external_params.update(cli_params)

    # Params are validated against the lowered program, so this preflight lowers
    # the graph.  It must report a param failure (exit 1) BEFORE the trace file
    # is prepared and the runner is built — hence a check-only pass here rather
    # than letting the run below surface it.  The lowered program it produces is
    # handed to that run, so the graph is lowered exactly once per invocation.
    param_preflight = runtime.preflight_params(
        prepared,
        param_values=external_params,
        compiled=discovery.compiled,
    )
    if not param_preflight.result.ok:
        for diag in param_preflight.result.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

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

    program_symbol = None
    if selected_program is not None:
        assert param_preflight.executable is not None
        program_symbol = param_preflight.executable.program_symbols[selected_program.node_id]

    # Reuse the ``PreparedProgram`` from above — no second parse/scope of the source.
    # Pass the already-computed compiled from discovery and the program the
    # preflight already lowered, so the graph is type-checked, match-compiled and
    # lowered exactly once.
    result = runtime.run_prepared(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        compiled=discovery.compiled,
        executable=param_preflight.executable,
        host_settings_policy=policy,
        builtin_host_settings=engine_seeds.values,
        program_symbol=program_symbol,
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
    param_tokens: list[str],
    *,
    args: ExecArgs | None = None,
    package: str | None = None,
    command_path: str | None = None,
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
        command = target.package.manifest.commands.get(command_path)
        if target.module_id.segments[0] != package or command is None:
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
            if target.module_id.segments[0] != package:
                _registered_command_mismatch(command_path)

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
            param_tokens=param_tokens,
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
        ),
        entry_module_segments=target.module_id.segments,
    )
