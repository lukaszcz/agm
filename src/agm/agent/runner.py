"""Shared helpers for running prompt-driven agent commands."""

from __future__ import annotations

import shlex
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.agent.prompt import expand_prompt_env_vars, preprocess_prompt_file
from agm.core import dry_run
from agm.core.fs import is_file
from agm.core.process import ProcessCaptureResult, run_capture, run_capture_result


@dataclass(slots=True)
class ResolvedPrompt:
    """Resolved prompt source: either inline text or a file path."""

    source: str | Path
    effective_file: Path


@dataclass(slots=True)
class PreparedPromptRun:
    """Prepared agent prompt command and prompt files."""

    command: list[str]
    effective_file: Path
    env: dict[str, str]
    temp_files: list[Path]


@dataclass(slots=True)
class PromptRunResult:
    """Structured result of a runner-backed agent prompt run.

    Unlike the existing ``run_prompt_command`` helper, this never prints to
    stderr and never raises ``SystemExit``.  All outcomes — spawn failure,
    nonzero exit, and idle-timeout — are represented here so that callers can
    map them to structured AgL exceptions (``AgentCallError``).
    """

    returncode: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool
    spawn_error: str | None


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


def prepare_prompt_run(
    *,
    runner: str,
    prompt_source: str | Path,
    extra_prompt_source: str | Path | None,
    env: dict[str, str],
    temp_files: list[Path],
    kind: str,
) -> PreparedPromptRun:
    """Prepare command and prompt files for a prompt-driven agent invocation."""

    command = split_command(runner, kind=kind)
    validate_command(command, kind=kind)
    resolved = prepare_prompt_from_source(prompt_source, temp_files=temp_files, env=env)
    effective_file = resolved.effective_file
    if extra_prompt_source is not None:
        effective_file = append_extra_prompt(
            effective_file,
            extra_prompt_source,
            temp_files=temp_files,
            env=env,
        )
    return PreparedPromptRun(
        command=command,
        effective_file=effective_file,
        env=env,
        temp_files=temp_files,
    )


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


def run_prepared_prompt(
    prepared: PreparedPromptRun,
    *,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> str:
    """Run a prepared prompt invocation."""

    if dry_run.enabled():
        dry_run.print_labeled_command(
            "agent",
            command_with_prompt_target(prepared.command, prepared.effective_file),
        )
        return ""
    return run_prompt_command(
        prepared.command,
        prepared.effective_file,
        env=prepared.env,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )


def cleanup_temp_files(temp_files: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass


def prepare_rendered_prompt_run(
    rendered_prompt: str,
    *,
    runner: str,
    temp_files: list[Path],
    env: dict[str, str],
) -> PreparedPromptRun:
    """Prepare a runner invocation for an already-rendered AgL prompt.

    Writes *rendered_prompt* verbatim to a temporary file.  Crucially:

    - Does **not** call ``expand_prompt_env_vars``: AgL interpolation has
      already produced the final text and interpolated values may legitimately
      contain ``$NAME``/``${NAME}`` syntax (plan §9.5).
    - Does **not** call ``validate_command``: that helper prints to stderr and
      raises ``SystemExit``, bypassing the ``AgentCallError`` structured path.
      Executable-not-found is instead represented in the ``PromptRunResult``
      returned by ``run_prepared_prompt_result``.
    """
    command = split_command(runner, kind="exec-runner")
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(rendered_prompt)
        temp_path = Path(handle.name)
    temp_files.append(temp_path)
    return PreparedPromptRun(
        command=command,
        effective_file=temp_path,
        env=env,
        temp_files=temp_files,
    )


def run_prepared_prompt_result(
    prepared: PreparedPromptRun,
    *,
    idle_timeout: float | None,
) -> PromptRunResult:
    """Run a prepared runner invocation and return a structured result.

    Unlike ``run_prepared_prompt`` / ``run_prompt_command``, this function
    **never prints to stderr** and **never raises SystemExit**.  All outcomes
    are represented in the returned :class:`PromptRunResult`.
    """
    capture: ProcessCaptureResult = run_capture_result(
        command_with_prompt_target(prepared.command, prepared.effective_file),
        env=prepared.env if prepared.env else None,
        idle_timeout=idle_timeout,
        isolate_process_group=True,
    )
    return PromptRunResult(
        returncode=capture.returncode,
        stdout=capture.stdout,
        stderr=capture.stderr,
        elapsed=capture.elapsed,
        timed_out=capture.timed_out,
        spawn_error=capture.spawn_error,
    )
