"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source file (exit 1 if unreadable), load the
``[exec]`` configuration, construct a ``WorkflowRuntime`` with the resolved
settings, call ``runtime.run`` (or a static-only dry run under ``--dry-run``),
print diagnostics to stderr, and exit per the exit-code contract.

Warning-severity diagnostics are printed to stderr like errors but never affect
the exit code; only error-severity diagnostics yield exit 1.

Exit-code contract (plan §10.1):
    0  success (or a clean ``--dry-run`` static check)
    1  pre-execution failure (unreadable file, static errors, input validation)
    2  program executed but ended with an uncaught AgL exception

Flag notes:
    - ``--strict-json`` controls JSON-codec strictness: when set, agents must
      return exactly one bare JSON value; the default is lenient recovery
      (fence/prose stripping + trivial repair, then strict schema validation).
      A source-level ``strict_json`` call option overrides this default.
    - ``--runner`` and ``--log-file``/``--no-log`` are accepted but inert until
      the runner-backed default agent and trace logging land in M5.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agl import WorkflowRuntime
from agm.commands.args import ExecArgs
from agm.config.context import current_config_context
from agm.config.general import load_exec_config
from agm.core import dry_run
from agm.core.cli_helpers import parse_inputs
from agm.core.fs import read_text_arg


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    source = read_text_arg(Path(args.file))

    # Parse --input k=v pairs (validation: malformed pairs exit 1).
    try:
        inputs = parse_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Load [exec] configuration; CLI flags override config values.
    ctx = current_config_context()
    config = load_exec_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit

    runtime = WorkflowRuntime(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
    )

    # ``parse_inputs`` returns ``dict[str, str]``; ``run`` accepts a
    # ``Mapping[str, object]``, so no widening copy is needed.
    result = runtime.run(source, inputs=inputs, check_only=dry_run.enabled())

    # Warnings are reported but do not affect the exit code.
    warnings = [d for d in result.diagnostics if d.severity == "warning"]
    errors = [d for d in result.diagnostics if d.severity == "error"]

    for diag in warnings:
        print(f"line {diag.line}: {diag.message}", file=sys.stderr)

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
        for diag in errors:
            print(f"line {diag.line}: {diag.message}", file=sys.stderr)
        raise SystemExit(1)

    # Uncaught AgL exception: print and exit 2.
    message = result.error.fields.get("message")
    if isinstance(message, str) and message:
        print(f"AgL exception: {result.error.type_name}: {message}", file=sys.stderr)
    else:
        print(f"AgL exception: {result.error.type_name}", file=sys.stderr)
    raise SystemExit(2)
