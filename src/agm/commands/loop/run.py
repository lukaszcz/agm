"""agm loop run."""

from __future__ import annotations

from agm.cli_support.args import LoopArgs
from agm.core import dry_run
from agm.core.log import append_log

from . import step as step_command


def run(args: LoopArgs) -> None:
    runtime: step_command.LoopStepRuntime | None = None
    try:
        runtime = step_command.prepare_runtime(args)
        if dry_run.enabled():
            step_command.print_dry_run(runtime)
            return

        step_command.print_startup(runtime)
        step_number = 1
        while not step_command.execute_single_step(runtime, step_number=step_number):
            step_number += 1
    except KeyboardInterrupt:
        message = "\nInterrupted\n"
        print(message, end="")
        if runtime is not None:
            append_log(runtime.log_file, message)
        raise SystemExit(130)
    finally:
        if runtime is not None:
            step_command.cleanup_runtime(runtime)
