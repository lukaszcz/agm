"""Compatibility adapter for ``agm exec`` source selection.

The shared execution pipeline lives in :mod:`agm.commands.exec_program` so
registered package commands and explicit ``agm exec`` references use identical
parameter, configuration, engine-setting, and runtime handling.
"""

from __future__ import annotations

from pathlib import Path

from agm.agl import PipelineDriver
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.cli_support.args import ExecArgs
from agm.commands import exec_program
from agm.config.context import current_config_context
from agm.config.general import exec_config_from_merged


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
    # Keep these injectable seams on the established adapter for callers and
    # tests that customize host construction around the execution pipeline.
    exec_program.run(
        args,
        pipeline_factory=PipelineDriver,
        agent_factory=value_driven_agent_factory,
        config_context_loader=current_config_context,
        exec_config_loader=exec_config_from_merged,
    )
