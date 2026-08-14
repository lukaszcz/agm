"""Compatibility adapter for ``agm exec`` source selection.

The shared execution pipeline lives in :mod:`agm.commands.exec_program` so
registered package commands and explicit ``agm exec`` references use identical
parameter, configuration, engine-setting, and runtime handling.
"""

from __future__ import annotations

from pathlib import Path

from agm.cli_support.args import ExecArgs
from agm.commands import exec_program


def run(args: ExecArgs) -> None:
    """Run a file, inline source, or installed program reference."""
    if (
        args.file is not None
        and "::" in args.file
        and args.command is None
        and not Path(args.file).is_file()
    ):
        exec_program.run_registered(args.file, args.param_tokens, args=args)
        return
    exec_program.run(args)
