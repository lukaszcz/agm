"""agm loop."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

from agm.commands.args import LoopArgs
from agm.config.general import LoopConfig, load_loop_config
from agm.core import dry_run
from agm.core.env import agm_installation_prefix
from agm.core.fs import append_text, is_file, mkdir, unlink
from agm.core.process import run_capture
from agm.core.prompt import preprocess_prompt_file


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
        if is_file(candidate):
            return candidate
    return candidates[-1]


def _configured_loop_settings(command_name: str | None) -> LoopConfig:
    return load_loop_config(
        home=Path(os.environ["HOME"]),
        proj_dir=Path(os.environ["PROJ_DIR"]) if os.environ.get("PROJ_DIR") else None,
        cwd=Path.cwd(),
        command_name=command_name,
    )


def _step_header_text(step: int) -> str:
    return (
        "\n"
        "-------------------------------------------------------------\n"
        f"                        Step {step}\n"
        "-------------------------------------------------------------\n"
        "\n"
    )


def _selected_task_text(task_file: Path) -> str:
    return f"Selected task: {task_file}\n"


def _split_command(command: str, *, kind: str) -> list[str]:
    split_command = shlex.split(command)
    if split_command:
        return split_command
    print(f"Error: loop {kind} command is empty.", file=sys.stderr)
    raise SystemExit(1)


def _runner_command(args: LoopArgs) -> list[str]:
    configured = _configured_loop_settings(args.command_name)
    runner = args.runner if args.runner is not None else configured.runner
    selected = runner if runner is not None else "claude -p"
    return [*_split_command(selected, kind="runner"), *args.runner_args]


def _selector_command(args: LoopArgs) -> list[str] | None:
    configured = _configured_loop_settings(args.command_name)
    selector = args.selector if args.selector is not None else configured.selector
    if selector is None:
        return None
    return _split_command(selector, kind="selector")


def _tasks_dir(args: LoopArgs) -> Path:
    configured_tasks_dir = _configured_loop_settings(args.command_name).tasks_dir
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


def _validate_command(command: list[str], *, kind: str) -> None:
    if shutil.which(command[0]) is None:
        print(
            f"Error: {kind} command {command[0]} is not installed or not in PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _command_with_prompt_target(command: list[str], target: Path) -> list[str]:
    prompt_path = str(target)
    placeholders = ("%%", "%{PROMPT_FILE}")
    replaced_command: list[str] = []
    replaced = False

    for arg in command:
        updated = arg
        for placeholder in placeholders:
            if placeholder in updated:
                updated = updated.replace(placeholder, prompt_path)
                replaced = True
        replaced_command.append(updated)

    if replaced:
        return replaced_command
    return [*command, f"@{target}"]


def _loop_env(tasks_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TASKS_DIR"] = str(tasks_dir)
    return env


def _run_command(command: list[str], target: Path, *, env: dict[str, str]) -> str:
    _, stdout, stderr = run_capture(
        _command_with_prompt_target(command, target),
        env=env,
    )
    output = stdout
    if stderr:
        output += stderr
    return output


def _append_log(log_file: Path | None, header: str, output: str) -> None:
    if log_file is None:
        return
    append_text(log_file, header + output, encoding="utf-8")


def _print_output(output: str) -> None:
    if output:
        print(output, end="")


def _selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
    selected = output.strip()
    if not selected:
        return ""
    if "".join(selected.split()) == "COMPLETE":
        return None

    task_path = Path(selected)
    if not task_path.is_absolute():
        task_path = tasks_dir / task_path
    if not is_file(task_path):
        return selected
    return task_path


def _cleanup_temp_files(temp_files: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            unlink(temp_file)
        except FileNotFoundError:
            pass


def run(args: LoopArgs) -> None:
    temp_files: list[Path] = []
    tasks_dir = _tasks_dir(args)
    loop_env = _loop_env(tasks_dir)
    runner_command = _runner_command(args)
    selector_command = _selector_command(args)
    _validate_command(runner_command, kind="runner")
    if selector_command is not None:
        _validate_command(selector_command, kind="selector")

    try:
        prompt_file: Path | None = None
        if selector_command is None:
            prompt_file = _prompt_file("loop.md")
            if not is_file(prompt_file):
                print(f"Error: prompt file not found: {prompt_file}", file=sys.stderr)
                raise SystemExit(1)
            prompt_file = preprocess_prompt_file(prompt_file, temp_files=temp_files, env=loop_env)

        progress_file = _progress_file(args)
        if selector_command is None and not is_file(progress_file):
            bootstrap_prompt_file = _prompt_file("update_progress.md")
            if not is_file(bootstrap_prompt_file):
                print(f"Error: prompt file not found: {bootstrap_prompt_file}", file=sys.stderr)
                raise SystemExit(1)
            bootstrap_prompt_file = preprocess_prompt_file(
                bootstrap_prompt_file, temp_files=temp_files, env=loop_env
            )
            _run_command(runner_command, bootstrap_prompt_file, env=loop_env)

        selector_prompt_file: Path | None = None
        if selector_command is not None:
            selector_prompt_file = _prompt_file("update_progress.md")
            if not is_file(selector_prompt_file):
                print(f"Error: prompt file not found: {selector_prompt_file}", file=sys.stderr)
                raise SystemExit(1)
            selector_prompt_file = preprocess_prompt_file(
                selector_prompt_file, temp_files=temp_files, env=loop_env
            )

        log_file = _log_file(args)
        if log_file is not None:
            print(
                f"Logging to {log_file if args.log_file is not None else log_file.name}"
            )
            mkdir(log_file.parent, parents=True, exist_ok=True)

        if dry_run.enabled():
            if selector_command is None:
                target = prompt_file if prompt_file is not None else _prompt_file("loop.md")
                dry_run.print_command(_command_with_prompt_target(runner_command, target))
            else:
                assert selector_prompt_file is not None
                selector_args = _command_with_prompt_target(
                    selector_command,
                    selector_prompt_file,
                )
                dry_run.print_command(selector_args)
                dry_run.print_operation(
                    "loop-runner",
                    "subsequent runner invocations depend on selector output",
                )
            return

        step = 1
        while True:
            header = _step_header_text(step)
            print(header, end="")
            step += 1

            if selector_command is None:
                assert prompt_file is not None
                output = _run_command(runner_command, prompt_file, env=loop_env)
                _append_log(log_file, header, output)
                _print_output(output)
                if "".join(output.split()) == "COMPLETE":
                    print("\nCompleted.")
                    break
                continue

            assert selector_prompt_file is not None
            selector_outputs: list[str] = []
            while True:
                selector_output = _run_command(selector_command, selector_prompt_file, env=loop_env)
                selector_outputs.append(selector_output)

                next_task = _selector_result(selector_output, tasks_dir=tasks_dir)
                if next_task is None:
                    combined_selector_output = "".join(selector_outputs)
                    _append_log(log_file, header, combined_selector_output)
                    _print_output(combined_selector_output)
                    print("\nCompleted.")
                    return
                if isinstance(next_task, Path):
                    break

            selected_task_output = _selected_task_text(next_task)
            selector_transcript = "".join(selector_outputs)
            _append_log(log_file, header, selected_task_output + selector_transcript)
            _print_output(selector_transcript + "\n" + selected_task_output)
            runner_output = _run_command(runner_command, next_task, env=loop_env)
            _append_log(log_file, "", runner_output)
            _print_output(runner_output)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        _cleanup_temp_files(temp_files)
