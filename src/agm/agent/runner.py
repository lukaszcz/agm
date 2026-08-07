"""Shared helpers for running prompt-driven agent commands."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from collections import ChainMap
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.agent.prompt import (
    expand_prompt_env_vars,
    preprocess_prompt_file,
    require_prompt_file,
)
from agm.core import dry_run
from agm.core.process import ProcessCaptureResult, run_capture, run_capture_result
from agm.util.interp import (
    Hole,
    InterpolationError,
    Literal,
    Segment,
    interp_segments,
    split_template,
)

# Shell convention: exit code 127 means the command could not be found or
# executed.  Treated as a fatal runner-configuration error (see
# :func:`run_prompt_command`) so the loop does not retry a missing runner.
_RUNNER_NOT_FOUND_EXIT = 127

# The variable a runner command uses to name the prepared prompt file, and its
# shorthand alias.
PROMPT_FILE_VAR = "PROMPT_FILE"
PROMPT_FILE_ALIAS = "%%"


@dataclass(slots=True)
class ResolvedPrompt:
    """Resolved prompt source: either inline text or a file path."""

    source: str | Path
    effective_file: Path


@dataclass(slots=True)
class PreparedPromptRun:
    """Prepared agent prompt command and prompt files.

    ``prompt_via_stdin`` marks a spec whose backend reads the prompt from
    standard input rather than from an interpolated placeholder or an
    appended ``@<path>`` argument (see ``AgentCodex``); it is ``False`` for
    every other spec, which keep the existing file-based delivery.
    """

    command: list[str]
    effective_file: Path
    env: dict[str, str]
    temp_files: list[Path]
    prompt_via_stdin: bool = False


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


def parse_command(command: str, *, kind: str) -> list[str]:
    """Split a command, raising ``ValueError`` without host-process side effects."""
    try:
        split = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"invalid {kind} command: {exc}") from exc
    if not split:
        raise ValueError(f"{kind} command is empty")
    return split


def split_command(command: str, *, kind: str) -> list[str]:
    try:
        return parse_command(command, kind=kind)
    except ValueError as exc:
        print(f"Error: {exc}.", file=sys.stderr)
        raise SystemExit(1) from exc


def _split_command_element(text: str) -> list[Segment]:
    """Split a command element, expanding the ``%%`` alias into a real hole.

    Resolving the alias during the split rather than substituting it into the
    rendered output keeps it subject to the same variable lookup as
    ``%{PROMPT_FILE}``, and keeps an interpolated value that happens to contain
    ``%%`` from being rewritten.
    """
    segments: list[Segment] = []
    for segment in split_template(text):
        if isinstance(segment, Hole):
            segments.append(segment)
            continue
        for index, part in enumerate(segment.text.split(PROMPT_FILE_ALIAS)):
            if index:
                segments.append(Hole(PROMPT_FILE_VAR))
            if part:
                segments.append(Literal(part))
    return segments


def _targets_prompt_file(segments: list[Segment]) -> bool:
    return any(
        isinstance(segment, Hole) and segment.name == PROMPT_FILE_VAR for segment in segments
    )


def validate_command(command: list[str], *, kind: str, env: MutableMapping[str, str]) -> None:
    """Preflight-check a runner/selector command.

    *env* must be the same mapping the command will eventually be run with
    (see ``command_with_prompt_target``), so a hole that resolves at run time
    (e.g. ``%{TASKS_DIR}``) does not spuriously fail here.
    """
    what = f"{kind} command executable {command[0]!r}"
    variables = ChainMap({PROMPT_FILE_VAR: str(Path(""))}, env)
    try:
        segments = _split_command_element(command[0])
        # The prompt file does not exist yet; binding a placeholder still validates
        # every other hole strictly, but leaves the executable unresolvable.
        executable = interp_segments(segments, variables)
    except InterpolationError as exc:
        print(f"Error: cannot interpolate {what}: {exc}.", file=sys.stderr)
        raise SystemExit(1) from exc
    if _targets_prompt_file(segments):
        return
    if shutil.which(executable) is None:
        print(
            f"Error: {kind} command {executable} is not installed or not in PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _interpolate_command(
    command: list[str], target: Path, env: MutableMapping[str, str]
) -> tuple[list[str], bool]:
    """Interpolate *command* against *env*, binding ``PROMPT_FILE``/``%%`` to *target*.

    Returns the interpolated argv and whether any element targeted
    ``PROMPT_FILE``/``%%``. *env* must be the same mapping the child process
    will actually receive (see ``run_capture``'s ``env`` argument) so argv
    holes and the spawned process resolve names identically.
    """
    variables = ChainMap({PROMPT_FILE_VAR: str(target)}, env)
    interpolated: list[str] = []
    targeted = False

    for arg in command:
        try:
            segments = _split_command_element(arg)
            targeted = targeted or _targets_prompt_file(segments)
            interpolated.append(interp_segments(segments, variables))
        except InterpolationError as exc:
            exc.context = f"in command element {arg!r}"
            raise

    return interpolated, targeted


def command_with_prompt_target(
    command: list[str],
    target: Path,
    env: MutableMapping[str, str],
    *,
    append_target: bool = True,
) -> list[str]:
    """Interpolate *command* against *env*, binding ``PROMPT_FILE``/``%%`` to *target*.

    When no element targeted ``PROMPT_FILE``/``%%`` and *append_target* is
    true (the default), ``@<target>`` is appended so the backend still
    receives the prompt. A stdin-delivered spec passes ``append_target=False``
    so its argv is interpolated the same way but never gets an ``@<target>``
    argument — the prompt reaches the process on standard input instead.
    """
    interpolated, targeted = _interpolate_command(command, target, env)
    if targeted or not append_target:
        return interpolated
    return [*interpolated, f"@{target}"]


def command_with_prompt_target_or_exit(
    command: list[str], target: Path, env: MutableMapping[str, str]
) -> list[str]:
    """Attach a prompt target, reporting interpolation failures as CLI errors."""
    try:
        return command_with_prompt_target(command, target, env)
    except InterpolationError as exc:
        print(f"Error: cannot interpolate runner command: {exc}.", file=sys.stderr)
        raise SystemExit(1) from exc


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
    require_prompt_file(source_path)
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
        require_prompt_file(extra_path, label="extra prompt")
        extra_content = expand_prompt_env_vars(
            extra_path.read_text(encoding="utf-8"), env=env, source=extra_path
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
    validate_command(command, kind=kind, env=env)
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

    try:
        returncode, stdout, stderr = run_capture(
            command_with_prompt_target_or_exit(command, target, env),
            env=env,
            stdout_callback=handle_stdout,
            stderr_callback=handle_stderr,
            isolate_process_group=True,
            idle_timeout=idle_timeout,
        )
    except OSError as exc:
        # The executable-not-found/not-in-PATH check in ``validate_command`` is
        # deferred for prompt-targeting commands (the prompt file does not exist
        # at preflight time), so a spawn failure can still surface here — report
        # it the same clean way rather than letting the raw OSError escape.
        runner_name = command[0] if command else ""
        print(
            f"Error: runner command {runner_name!r} could not be run: {exc}.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Exit code 127 is the shell's "command not found" convention: the resolved
    # runner command could not be found or executed.  This is a fatal
    # configuration error, not a transient runner failure the loop should
    # retry — surface it immediately so ``agm loop``/``review``/``revise``/
    # ``refine`` break out instead of looping forever on a missing runner.
    if returncode == _RUNNER_NOT_FOUND_EXIT:
        runner_name = command[0] if command else ""
        msg = (
            f"Error: runner command {runner_name!r} could not be found or executed "
            f"(exit code {returncode})."
        )
        if stderr:
            stderr_tail = stderr if len(stderr) <= 500 else stderr[-500:]
            msg = f"{msg}\n{stderr_tail}"
        print(msg, file=sys.stderr)
        raise SystemExit(1)

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
            command_with_prompt_target_or_exit(
                prepared.command, prepared.effective_file, prepared.env
            ),
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
    runner: list[str],
    temp_files: list[Path],
    env: dict[str, str],
    prompt_via_stdin: bool = False,
) -> PreparedPromptRun:
    """Prepare a runner invocation for an already-rendered AgL prompt.

    Writes *rendered_prompt* verbatim to a temporary file.  Crucially:

    - Does **not** call ``expand_prompt_env_vars``: AgL interpolation has
      already produced the final text and interpolated values may legitimately
      contain ``$NAME``, ``${NAME}``, or ``%{name}`` syntax.
    - Does **not** call ``validate_command``: that helper prints to stderr and
      raises ``SystemExit``, bypassing the ``AgentCallError`` structured path.
      Executable-not-found is instead represented in the ``PromptRunResult``
      returned by ``run_prepared_prompt_result``.
    - Accepts an already-tokenized argv from an agent command builder, avoiding
      a string round-trip before the prepared invocation is run.

    *prompt_via_stdin* is carried onto the returned ``PreparedPromptRun`` so
    ``run_prepared_prompt_result`` knows to pipe the prompt file's contents in
    rather than attach it via placeholder or ``@<path>``.
    """
    command = runner.copy()
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(rendered_prompt)
        temp_path = Path(handle.name)
    temp_files.append(temp_path)
    return PreparedPromptRun(
        command=command,
        effective_file=temp_path,
        env=env,
        temp_files=temp_files,
        prompt_via_stdin=prompt_via_stdin,
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

    When ``prepared.prompt_via_stdin`` is set, the prompt file's contents are
    piped in as ``stdin_text`` instead of being attached via placeholder or
    ``@<path>``.
    """
    # An empty ``prepared.env`` means the child inherits ``os.environ`` (see the
    # ``env=None`` passed to ``run_capture_result`` below); interpolate argv
    # holes against the same effective mapping so both agree on variable values.
    child_env = prepared.env if prepared.env else os.environ
    stdin_text = None
    argv = command_with_prompt_target(
        prepared.command,
        prepared.effective_file,
        child_env,
        append_target=not prepared.prompt_via_stdin,
    )
    if prepared.prompt_via_stdin:
        stdin_text = prepared.effective_file.read_text(encoding="utf-8")
    capture: ProcessCaptureResult = run_capture_result(
        argv,
        env=prepared.env if prepared.env else None,
        stdin_text=stdin_text,
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
