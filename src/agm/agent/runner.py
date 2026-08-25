"""Shared helpers for running prompt-driven agent commands."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from collections import ChainMap
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.agent.prompt import (
    expand_prompt_env_vars,
    preprocess_prompt_file,
    require_prompt_file,
)
from agm.agent.transport import AgentTransportFailureCause
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
SESSION_ID_VAR = "SESSION_ID"
PROMPT_FILE_ALIAS = "%%"


class AgentCallTimeout(Exception):
    """An agent invocation exceeded its configured idle timeout."""

    def __init__(self, idle_timeout: float | None) -> None:
        self.idle_timeout = idle_timeout
        super().__init__(f"agent call timed out after {idle_timeout}s of inactivity")


@dataclass(slots=True)
class ResolvedPrompt:
    """Resolved prompt source: either inline text or a file path."""

    source: str | Path
    effective_file: Path


class PromptDelivery(StrEnum):
    """How a prepared agent invocation receives its rendered prompt."""

    FILE = "file"
    STDIN = "stdin"
    LITERAL = "literal"
    NONE = "none"


@dataclass(slots=True)
class PreparedPromptRun:
    """Prepared agent argv and any temporary prompt files.

    Most agent calls attach a temporary prompt file. Native lifecycle calls
    instead need either a literal argv argument (Claude's ``/compact``) or no
    prompt at all (forking); Codex receives its prompt on stdin. Keeping that
    choice explicit prevents a lifecycle command from accidentally receiving
    a file attachment.
    """

    command: list[str]
    effective_file: Path
    env: dict[str, str]
    temp_files: list[Path]
    stdin_prompt: str | None = None
    argv: list[str] | None = None
    delivery: PromptDelivery = PromptDelivery.FILE

    @property
    def prompt_via_stdin(self) -> bool:
        """Whether the backend receives the prompt on stdin rather than from a file."""
        return self.delivery is PromptDelivery.STDIN or self.stdin_prompt is not None


@dataclass(slots=True)
class PromptRunResult:
    """Structured result of a runner-backed agent prompt run.

    Unlike the existing ``run_prompt_command`` helper, this never prints to
    stderr and never raises ``SystemExit``.  All outcomes — spawn failure,
    nonzero exit, and idle-timeout — are represented here so that callers can
    map them to their own host errors.
    """

    returncode: int | None
    stdout: str
    stderr: str
    elapsed: float
    timed_out: bool
    spawn_error: str | None


class PromptRunFailure(Exception):
    """A failed structured prompt run, independent of any caller's error model."""

    def __init__(self, cause: AgentTransportFailureCause, result: PromptRunResult) -> None:
        self.cause = cause
        self.result = result
        super().__init__(_prompt_run_failure_message(cause, result))


def prompt_run_result_error(result: PromptRunResult) -> PromptRunFailure | None:
    """Return the failure represented by *result*, or ``None`` for success."""
    if result.spawn_error is not None:
        return PromptRunFailure("spawn_failure", result)
    if result.timed_out:
        return PromptRunFailure("timeout", result)
    if result.returncode not in (None, 0):
        return PromptRunFailure("nonzero_exit", result)
    return None


def _prompt_run_failure_message(cause: AgentTransportFailureCause, result: PromptRunResult) -> str:
    if cause == "spawn_failure":
        return f"agent command could not be started: {result.spawn_error}"
    if cause == "timeout":
        return "agent command timed out"
    return f"agent command exited with code {result.returncode}"


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


def command_targets_session_id(command: list[str]) -> bool:
    """Whether a command contains an unescaped session-id placeholder.

    Every element is parsed even after finding a placeholder so malformed
    interpolation is always reported while a command session is opened.
    """
    targets_session_id = False
    for arg in command:
        try:
            segments = _split_command_element(arg)
        except InterpolationError as exc:
            exc.context = f"in command element {arg!r}"
            raise
        targets_session_id = targets_session_id or any(
            isinstance(segment, Hole) and segment.name == SESSION_ID_VAR for segment in segments
        )
    return targets_session_id


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
    command: list[str],
    target: Path,
    env: MutableMapping[str, str],
    *,
    session_id: str | None = None,
) -> tuple[list[str], bool]:
    """Interpolate *command* against *env*, binding prompt and session placeholders.

    ``PROMPT_FILE``/``%%`` always resolve to *target*. When *session_id* is
    supplied, ``SESSION_ID`` resolves to it; otherwise it retains the normal
    environment-backed interpolation behavior. Returns the interpolated argv
    and whether any element targeted ``PROMPT_FILE``/``%%``. *env* must be the
    same mapping the child process will actually receive (see
    ``run_capture``'s ``env`` argument) so argv holes and the spawned process
    resolve names identically.
    """
    bindings = {PROMPT_FILE_VAR: str(target)}
    if session_id is not None:
        bindings[SESSION_ID_VAR] = session_id
    variables = ChainMap(bindings, env)
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
    session_id: str | None = None,
) -> list[str]:
    """Interpolate *command* against *env*, binding ``PROMPT_FILE``/``%%`` to *target*.

    When no element targeted ``PROMPT_FILE``/``%%`` and *append_target* is
    true (the default), ``@<target>`` is appended so the backend still
    receives the prompt. A stdin-delivered spec passes ``append_target=False``
    so its argv is interpolated the same way but never gets an ``@<target>``
    argument — the prompt reaches the process on standard input instead.
    """
    interpolated, targeted = _interpolate_command(command, target, env, session_id=session_id)
    if targeted or not append_target:
        return interpolated
    return [*interpolated, f"@{target}"]


def command_with_prompt_target_or_exit(
    command: list[str],
    target: Path,
    env: MutableMapping[str, str],
    *,
    append_target: bool = True,
) -> list[str]:
    """Attach a prompt target, reporting interpolation failures as CLI errors."""
    try:
        return command_with_prompt_target(command, target, env, append_target=append_target)
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
    stdin_text: str | None = None,
    prepared_argv: list[str] | None = None,
    append_target: bool = True,
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

    def handle_timeout(message: str) -> None:
        handle_stderr(message)
        if stderr_callback is None:
            print(message, end="", file=sys.stderr)

    argv = (
        command_with_prompt_target_or_exit(command, target, env, append_target=append_target)
        if prepared_argv is None
        else prepared_argv
    )
    try:
        if stdin_text is None:
            returncode, stdout, stderr = run_capture(
                argv,
                env=env,
                stdout_callback=handle_stdout,
                stderr_callback=handle_stderr,
                timeout_callback=handle_timeout,
                isolate_process_group=True,
                idle_timeout=idle_timeout,
            )
        else:
            returncode, stdout, stderr = run_capture(
                argv,
                env=env,
                stdout_callback=handle_stdout,
                stderr_callback=handle_stderr,
                timeout_callback=handle_timeout,
                isolate_process_group=True,
                idle_timeout=idle_timeout,
                stdin_text=stdin_text,
            )
    except SystemExit as exc:
        # ``run_capture`` predates structured process results and represents an
        # idle timeout as SystemExit(124). At the agent boundary this is only a
        # failed invocation; callers such as ``agm loop`` decide whether to retry.
        if exc.code == 124:
            raise AgentCallTimeout(idle_timeout) from exc
        raise
    except OSError as exc:
        # Prompt-targeting commands defer preflight validation because the prompt
        # file does not exist yet, so a spawn failure can still surface here.
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
        handle_stderr(msg + "\n")
        if stderr_callback is None:
            print(msg, file=sys.stderr)
        raise SystemExit(1)

    if ordered_output:
        return "".join(ordered_output)
    output = stdout
    if stderr:
        output += stderr
    return output


def _prepared_stdin_text(prepared: PreparedPromptRun) -> str | None:
    """Return piped input, using an empty pipe to deliver EOF explicitly."""
    return "" if prepared.delivery is PromptDelivery.NONE else prepared.stdin_prompt


def run_prepared_prompt(
    prepared: PreparedPromptRun,
    *,
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
) -> str:
    """Run a prepared prompt invocation."""

    append_target = prepared.delivery is PromptDelivery.FILE and not prepared.prompt_via_stdin
    if dry_run.enabled():
        dry_run.print_labeled_command(
            "agent",
            prepared.argv
            if prepared.argv is not None
            else command_with_prompt_target_or_exit(
                prepared.command,
                prepared.effective_file,
                prepared.env,
                append_target=append_target,
            ),
        )
        return ""
    if prepared.argv is None and prepared.stdin_prompt is None and append_target:
        return run_prompt_command(
            prepared.command,
            prepared.effective_file,
            env=prepared.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
        )
    return run_prompt_command(
        prepared.command,
        prepared.effective_file,
        env=prepared.env,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        stdin_text=_prepared_stdin_text(prepared),
        prepared_argv=prepared.argv,
        append_target=append_target,
    )


def cleanup_temp_files(temp_files: list[Path]) -> None:
    """Remove temporary prompt files unless dry-run keeps them for inspection."""
    if dry_run.enabled():
        return
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
    prompt_via_stdin: bool | None = None,
    delivery: PromptDelivery | None = None,
    session_id: str | None = None,
) -> PreparedPromptRun:
    """Prepare a runner invocation for an already-rendered AgL prompt.

    Crucially:

    - Does **not** call ``expand_prompt_env_vars``: AgL interpolation has
      already produced the final text and interpolated values may legitimately
      contain ``$NAME``, ``${NAME}``, or ``%{name}`` syntax.
    - Does **not** call ``validate_command``: that helper prints to stderr and
      raises ``SystemExit``, bypassing the ``AgentCallError`` structured path.
      Executable-not-found is instead represented in the ``PromptRunResult``
      returned by ``run_prepared_prompt_result``.
    - Accepts an already-tokenized argv from an agent command builder, avoiding
      a string round-trip before the prepared invocation is run.
    - Binds ``%{SESSION_ID}`` when *session_id* is provided, without changing
      ordinary runner interpolation when it is not.

    ``prompt_via_stdin`` remains the compatibility spelling for stdin
    delivery. New callers use *delivery* so literal and promptless lifecycle
    commands share this same subprocess boundary without making prompt files.
    """
    if delivery is None:
        delivery = PromptDelivery.STDIN if prompt_via_stdin else PromptDelivery.FILE
    elif prompt_via_stdin:
        raise ValueError("prompt_via_stdin cannot be combined with an explicit delivery")

    command = runner.copy()
    child_env = env if env else os.environ
    if delivery is PromptDelivery.FILE:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
            handle.write(rendered_prompt)
            effective_file = Path(handle.name)
        temp_files.append(effective_file)
        argv = command_with_prompt_target(
            command,
            effective_file,
            child_env,
            append_target=True,
            session_id=session_id,
        )
        return PreparedPromptRun(
            command=command,
            effective_file=effective_file,
            env=env,
            temp_files=temp_files,
            argv=argv,
            delivery=delivery,
        )

    effective_file = Path(os.devnull)
    argv = command_with_prompt_target(
        command,
        effective_file,
        child_env,
        append_target=False,
        session_id=session_id,
    )
    if delivery is PromptDelivery.LITERAL:
        argv.append(rendered_prompt)
    return PreparedPromptRun(
        command=command,
        effective_file=effective_file,
        env=env,
        temp_files=temp_files,
        stdin_prompt=rendered_prompt if delivery is PromptDelivery.STDIN else None,
        argv=argv,
        delivery=delivery,
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

    When ``prepared.stdin_prompt`` is set, it is piped in as ``stdin_text``
    instead of being attached via placeholder or ``@<path>`` — it is
    delivered directly from the already-rendered text rather than read back
    off disk.
    """
    # An empty ``prepared.env`` means the child inherits ``os.environ`` (see the
    # ``env=None`` passed to ``run_capture_result`` below); interpolate argv
    # holes against the same effective mapping so both agree on variable values.
    child_env = prepared.env if prepared.env else os.environ
    argv = prepared.argv or command_with_prompt_target(
        prepared.command,
        prepared.effective_file,
        child_env,
        append_target=not prepared.prompt_via_stdin,
    )
    capture: ProcessCaptureResult = run_capture_result(
        argv,
        env=prepared.env if prepared.env else None,
        stdin_text=_prepared_stdin_text(prepared),
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
