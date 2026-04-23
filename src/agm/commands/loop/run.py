"""agm loop run."""

from __future__ import annotations

from agm.commands.args import LoopArgs
from agm.core import dry_run

from . import step as step_command


def run(args: LoopArgs) -> None:
    runtime: step_command.LoopStepRuntime | None = None
    try:
        runtime = step_command.prepare_runtime(args)
        if dry_run.enabled():
            step_command.print_dry_run(runtime)
            return

        step_number = 1
        while not step_command.execute_single_step(runtime, step_number=step_number):
            step_number += 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        if runtime is not None:
            step_command.cleanup_runtime(runtime)
