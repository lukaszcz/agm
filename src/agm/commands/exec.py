"""Compatibility adapter for ``agm exec`` source selection.

The shared execution pipeline lives in :mod:`agm.commands.exec_program` so
registered package commands and explicit ``agm exec`` references use identical
parameter, configuration, engine-setting, and runtime handling.
"""

from __future__ import annotations

from agm.cli_support.args import ExecArgs
from agm.cli_support.exec_target import is_installed_reference
from agm.cli_support.program_options import (
    ProgramHelpRequested,
    exec_program_name,
    render_program_help,
)
from agm.commands import exec_program
from agm.commands.exec_program import RegisteredProgramUsageError
from agm.parser import exit_with_usage_error


def run(args: ExecArgs) -> None:
    """Run a file, inline source, or installed program reference.

    A help request the CLI layer did not already recognize — a ``-h`` bundled
    into a short group of the program's own — reaches the selected program's
    command here, and renders that command's help instead of running it.
    """
    try:
        if args.file is not None and is_installed_reference(args.file, command=args.command):
            exec_program.run_registered(args.file, args.argument_tokens, args=args)
            return
        exec_program.run(args)
    except ProgramHelpRequested as exc:
        print(
            render_program_help(
                exc.command,
                program_name=exec_program_name(file=args.file, program=args.program),
            ),
            end="",
        )
    except RegisteredProgramUsageError as exc:
        exit_with_usage_error(["exec"], f"error: {exc.message}")
