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
from agm.core.env import agm_installation_prefix


def _prompt_dir_candidates() -> list[Path]:
    candidates: list[Path] = []

    install_prefix = agm_installation_prefix()
    if install_prefix is not None:
        candidates.append(install_prefix / ".agm" / "prompts")

    home = Path(os.environ["HOME"])
    candidates.append(home / ".agm" / "prompts")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def _prompt_file(filename: str) -> Path:
    candidates = [prompt_dir / filename for prompt_dir in _prompt_dir_candidates()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _configured_loop_settings() -> tuple[str | None, str | None]:
    configured = load_loop_config(
        home=Path(os.environ["HOME"]),
        proj_dir=Path(os.environ["PROJ_DIR"]) if os.environ.get("PROJ_DIR") else None,
        cwd=Path.cwd(),
    )
    return configured.command, configured.tasks_dir


def _step_header_text(step: int) -> str:
    return (
        "\n"
        "-------------------------------------------------------------\n"
        f"                        Step {step}\n"
        "-------------------------------------------------------------\n"
        "\n"
    )


def _loop_command(args: LoopArgs) -> list[str]:
    configured_command, _configured_tasks_dir = _configured_loop_settings()
    command = args.command if args.command is not None else configured_command
    selected = "claude -p" if command is None else command
    return shlex.split(selected)


def _tasks_dir(args: LoopArgs) -> Path:
    _configured_command, configured_tasks_dir = _configured_loop_settings()
    selected = args.tasks_dir if args.tasks_dir is not None else configured_tasks_dir
    if selected is None:
        return Path.cwd() / ".agent-files" / "tasks"

    tasks_dir = Path(selected)
    if tasks_dir.is_absolute():
        return tasks_dir
    return Path.cwd() / tasks_dir


def _progress_file(args: LoopArgs) -> Path:
    return _tasks_dir(args) / "PROGRESS.md"


def _log_file(args: LoopArgs) -> Path | None:
    if args.no_log:
        return None
    if args.log_file is not None:
        return Path(args.log_file)
    return Path.cwd() / f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def run(args: LoopArgs) -> None:
    prompt_file = _prompt_file("loop.md")
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

    if not _progress_file(args).is_file():
        bootstrap_prompt_file = _prompt_file("update_progress.md")
        if not bootstrap_prompt_file.is_file():
            print(f"Error: prompt file not found: {bootstrap_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        subprocess.run(
            [*command, f"@{bootstrap_prompt_file}"],
            capture_output=True,
            text=True,
            check=False,
        )

    log_file = _log_file(args)
    if log_file is not None:
        print(
            f"Logging to {log_file if args.log_file is not None else log_file.name}"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

    step = 1
    try:
        while True:
            header = _step_header_text(step)
            print(header, end="")
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

            if log_file is not None:
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(header)
                    handle.write(output)

            if output:
                print(output, end="")

            if "".join(output.split()) == "COMPLETE":
                print("\nCompleted.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
