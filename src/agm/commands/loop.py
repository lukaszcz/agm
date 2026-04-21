"""agm loop."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from agm.commands.args import LoopArgs
from agm.config.general import load_loop_config


def _prompt_file_candidates() -> list[Path]:
    candidates: list[Path] = []

    agm_executable = shutil.which("agm")
    if agm_executable is not None:
        candidates.append(
            Path(agm_executable).resolve().parent.parent / ".agm" / "prompts" / "loop.md"
        )

    home = Path(os.environ["HOME"])
    candidates.append(home / ".agm" / "prompts" / "loop.md")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _prompt_file() -> Path:
    for candidate in _prompt_file_candidates():
        if candidate.is_file():
            return candidate
    return _prompt_file_candidates()[-1]


def _print_step_header(step: int) -> None:
    print()
    print("-------------------------------------------------------------")
    print(f"                        Step {step}")
    print("-------------------------------------------------------------")
    print()


def _loop_command(args: LoopArgs) -> list[str]:
    configured = load_loop_config(
        home=Path(os.environ["HOME"]),
        proj_dir=Path(os.environ["PROJ_DIR"]) if os.environ.get("PROJ_DIR") else None,
        cwd=Path.cwd(),
    ).command
    command = args.command if args.command is not None else configured
    selected = "claude -p" if command is None else command
    return shlex.split(selected)


def run(args: LoopArgs) -> None:
    prompt_file = _prompt_file()
    if not prompt_file.is_file():
        print(f"Error: prompt file not found: {prompt_file}", file=sys.stderr)
        raise SystemExit(1)

    command = _loop_command(args)
    if not command:
        print("Error: loop command is empty.", file=sys.stderr)
        raise SystemExit(1)

    if shutil.which(command[0]) is None:
        print(f"Error: {command[0]} is not installed or not in PATH.", file=sys.stderr)
        raise SystemExit(1)

    log_file = Path.cwd() / f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    print(f"Logging to {log_file.name}")

    step = 1
    try:
        while True:
            _print_step_header(step)
            step += 1

            result = subprocess.run(
                [*command, f"@{prompt_file}"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr

            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(output)

            if output:
                print(output, end="")

            if "".join(output.split()) == "COMPLETE":
                print("Completed.")
                break
    except KeyboardInterrupt:
        print("Interrupted")
        raise SystemExit(130)
