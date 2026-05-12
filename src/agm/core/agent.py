"""Shared helpers for running prompt-driven agent commands."""

from __future__ import annotations

import shlex
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.core.fs import is_file
from agm.core.process import run_capture
from agm.core.prompt import expand_prompt_env_vars, preprocess_prompt_file


@dataclass(slots=True)
class ResolvedPrompt:
    """Resolved prompt source: either inline text or a file path."""

    source: str | Path
    effective_file: Path


def split_command(command: str, *, kind: str) -> list[str]:
    split = shlex.split(command)
    if split:
        return split
    print(f"Error: {kind} command is empty.", file=sys.stderr)
    raise SystemExit(1)


def validate_command(command: list[str], *, kind: str) -> None:
    if shutil.which(command[0]) is None:
        print(
            f"Error: {kind} command {command[0]} is not installed or not in PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def command_with_prompt_target(command: list[str], target: Path) -> list[str]:
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


def prepare_prompt_from_source(
    source: str | Path,
    *,
    temp_files: list[Path],
    env: dict[str, str],
) -> ResolvedPrompt:
    """Create a preprocessed prompt file from inline text or a file path."""

    if isinstance(source, str):
        expanded = expand_prompt_env_vars(source, env=env)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
            handle.write(expanded)
            temp_path = Path(handle.name)
        temp_files.append(temp_path)
        return ResolvedPrompt(source=source, effective_file=temp_path)

    source_path = source
    if not is_file(source_path):
        print(
            f"Error: prompt file not found: {source_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    effective = preprocess_prompt_file(source_path, temp_files=temp_files, env=env)
    return ResolvedPrompt(source=source_path, effective_file=effective)


def append_extra_prompt(
    effective_file: Path,
    extra_source: str | Path,
    *,
    temp_files: list[Path],
    env: dict[str, str],
) -> Path:
    """Append env-expanded extra prompt content to an effective prompt file."""

    original_content = effective_file.read_text(encoding="utf-8")
    if isinstance(extra_source, str):
        extra_content = expand_prompt_env_vars(extra_source, env=env)
    else:
        extra_path = extra_source
        if not is_file(extra_path):
            print(
                f"Error: extra prompt file not found: {extra_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        extra_content = expand_prompt_env_vars(
            extra_path.read_text(encoding="utf-8"), env=env
        )
    combined = original_content + "\n" + extra_content
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(combined)
        new_path = Path(handle.name)
    temp_files.append(new_path)
    return new_path


def run_prompt_command(
    command: list[str],
    target: Path,
    *,
    env: dict[str, str],
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    idle_timeout: float | None = None,
) -> str:
    ordered_output: list[str] = []

    def handle_stdout(chunk: str) -> None:
        ordered_output.append(chunk)
        if stdout_callback is not None:
            stdout_callback(chunk)

    def handle_stderr(chunk: str) -> None:
        ordered_output.append(chunk)
        if stderr_callback is not None:
            stderr_callback(chunk)

    _, stdout, stderr = run_capture(
        command_with_prompt_target(command, target),
        env=env,
        stdout_callback=handle_stdout,
        stderr_callback=handle_stderr,
        isolate_process_group=True,
        idle_timeout=idle_timeout,
    )
    if ordered_output:
        return "".join(ordered_output)
    output = stdout
    if stderr:
        output += stderr
    return output


def cleanup_temp_files(temp_files: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
