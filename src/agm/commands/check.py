"""Implementation of the ``agm check FILE...`` command.

Behaviour: run the full **static** AgL pipeline (parse, module loading, scope
resolution, type checking, match compilation, and lowering) over each given
``.agl`` file, independently and in argument order, printing GNU-style
diagnostics to stderr. Unlike ``agm exec``, no file needs to declare a
``program def`` — library modules can be checked too. A declared ``program
def`` is validated as part of the module it lives in, but ``check`` never
selects, validates its arguments, or runs one: it never evaluates anything
and never invokes an agent. Every file is checked even when an earlier one
failed; exit code is 1 iff any file produced an error-severity diagnostic,
was unreadable/missing, or had an invalid module-root configuration.

Static path and reuse: this reuses exactly the same building blocks
``agm exec`` uses for module resolution and file reading
(``agm.cli_support.exec_roots.effective_exec_roots_or_none``, ``agm.core.fs.
read_text_arg_or_none``) and diagnostic rendering (``agm.agl.diagnostics.
format_diagnostic``) — never duplicated. Unlike ``exec``, ``check`` never
selects or runs a ``program def``, so it has no use for ``exec``'s
program-discovery/selection/preflight machinery
(``PipelineDriver.discover_programs``/``preflight_arguments``): the smallest
path that still reaches lowering without binding or validating any host
argument is ``PipelineDriver.parse_entry`` → ``PipelineDriver.
prepare_parsed_entry`` → ``PipelineDriver.check_prepared(...)`` directly.
Lowering (rather than stopping at match compilation, as the REPL's own
``--dry-run`` does) is required here because some static errors — an invalid
``resource``/``resource-dir`` path, an unmaterializable output contract —
surface only during contract materialization and lowering, not during type
checking. ``check_prepared`` reaches exactly that far and no further: it
never resolves or validates a program's arguments, so a required ``program
def`` value parameter with no default is silently accepted, unlike under
``agm exec --dry-run``.

Warnings (on ``RunResult.warnings``) are a separate channel from error
diagnostics, printed to stderr but never affecting the exit code — the same
channel discipline ``agm exec`` uses (see ``commands/exec_program.py``).

Exit-code contract:
    0  no file produced an error-severity diagnostic
    1  some file did, was unreadable/missing, or had an invalid module-root
       configuration
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.cli_support.args import CheckArgs
from agm.cli_support.exec_roots import effective_exec_roots_or_none
from agm.config.context import current_config_context
from agm.core.fs import read_text_arg_or_none


def run(args: CheckArgs) -> None:
    """Statically check each file in ``args.files``, independently, in order."""
    from agm.agl import PipelineDriver
    from agm.agl.diagnostics import format_diagnostic

    ctx = current_config_context()
    runtime = PipelineDriver(get_sandbox_context=None)
    had_error = False

    for file in args.files:
        entry_path = Path(file)
        source = read_text_arg_or_none(entry_path)
        if source is None:
            # ``read_text_arg_or_none`` already printed a consistent
            # "Error: cannot read ..." message; keep checking the remaining
            # files.
            had_error = True
            continue

        exec_roots = effective_exec_roots_or_none(
            entry_path=entry_path,
            module_paths=args.module_paths,
            cwd=ctx.cwd,
            home=ctx.home,
            proj_dir=ctx.proj_dir,
        )
        if exec_roots is None:
            # ``effective_exec_roots_or_none`` already printed the canonical
            # "Error: invalid module roots configuration ..." message.
            had_error = True
            continue

        parsed = PipelineDriver.parse_entry(source, entry_path=entry_path)
        prepared = PipelineDriver.prepare_parsed_entry(
            parsed,
            roots=exec_roots.roots,
            default_stdlib=not args.no_stdlib,
        )
        result = runtime.check_prepared(prepared)

        for diag in result.warnings:
            print(format_diagnostic(diag, source_name=file), file=sys.stderr)
        if not result.ok:
            for diag in result.diagnostics:
                print(format_diagnostic(diag, source_name=file), file=sys.stderr)
            had_error = True

    if had_error:
        raise SystemExit(1)
