"""Compatibility adapter for ``agm exec`` source selection.

The shared execution pipeline lives in :mod:`agm.commands.exec_program` so
registered package commands and explicit ``agm exec`` references use identical
parameter, configuration, engine-setting, and runtime handling.
"""

from __future__ import annotations

from agm.cli_support.args import ExecArgs
from agm.cli_support.exec_target import is_installed_reference
from agm.commands import exec_program
from agm.commands.exec_program import RegisteredProgramUsageError
from agm.parser import exit_with_usage_error


def run(args: ExecArgs) -> None:
    """Run a file, inline source, or installed program reference."""
    try:
        if args.file is not None and is_installed_reference(args.file, command=args.command):
            exec_program.run_registered(args.file, args.argument_tokens, args=args)
            return
        exec_program.run(args)
    except RegisteredProgramUsageError as exc:
        exit_with_usage_error(["exec"], f"error: {exc.message}")
