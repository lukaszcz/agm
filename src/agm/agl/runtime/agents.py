"""AgentRegistry for the AgL runtime.

``AgentRegistry`` holds the callable agents registered with a
``WorkflowRuntime``.  It distinguishes:

- **Named agents**: registered with ``register_agent(name, fn)``.
- **Default agent** (``prompt``): registered via the ``default_agent``
  constructor kwarg; handles the built-in ``prompt`` contextual keyword.  It
  also backs any *declared* named agent that has no dedicated registration:
  scope guarantees every named call is declared, so the default agent is the
  documented backing for a declared name, not an implicit resolver of arbitrary
  names.

The registry is also the source of the ``HostCapabilities.has_default_agent``
and ``HostCapabilities.agent_names`` values.

Runner-backed agent
-------------------
``runner_backed_agent_factory`` builds an ``AgentFn`` that dispatches agent
calls to an external runner process via ``agm.agent.runner``.  It composes the
message text sent to the process (rendered prompt + format instructions +
§7.8 corrective retry feedback) and maps subprocess failures to
``AgentCallHostError``.

AgentCallError surfacing seam
------------------------------
``AgentCallHostError`` is a Python exception raised by runner-backed agents
(and any other host-level transport failure) to signal that the subprocess
failed.  The ``AgentRegistry.dispatch`` method catches it and re-raises as
``AglRaise(ExceptionValue("AgentCallError", ...))`` so the AgL interpreter can
handle it as a catchable in-language exception.

This design was chosen because:
1. The registry IS owned by this milestone (M5a).
2. The interpreter (``eval/``) is off-limits for this agent.
3. Converting at the dispatch boundary is the right abstraction: the registry
   is the single chokepoint through which every agent call flows, so the
   conversion happens once for all call sites and agents, without duplicating
   conversion logic.
4. Circular-import concern is sidestepped via local imports inside the method
   (the same pattern already used throughout ``runtime.py``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from agm.agl.runtime.request import AgentRequest, AgentResponse

# A host agent callable may return a plain ``str`` or a full ``AgentResponse``.
AgentFn = Callable[[AgentRequest], AgentResponse | str]


# ---------------------------------------------------------------------------
# AgentCallHostError — raised by transport-level failures
# ---------------------------------------------------------------------------


class AgentCallHostError(Exception):
    """Python-level exception for runner/transport failures.

    Raised by runner-backed agents (and any other host agent that experiences a
    transport-level failure) to carry structured failure information.  The
    ``AgentRegistry.dispatch`` method converts this to an in-language
    ``AglRaise(ExceptionValue("AgentCallError", ...))`` so AgL programs can
    catch it.

    ``cause``
        One of ``"spawn_failure"``, ``"nonzero_exit"``, or ``"timeout"``.
    ``exit_code``
        The process exit code (``None`` for spawn failures).
    ``stderr_tail``
        The last portion of the process's stderr output.
    ``elapsed``
        Wall time elapsed during the call (seconds).
    """

    def __init__(
        self,
        *,
        cause: str,
        exit_code: int | None,
        stderr_tail: str,
        elapsed: float,
    ) -> None:
        super().__init__(cause)
        self.cause = cause
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail
        self.elapsed = elapsed


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Immutable-after-build registry of host agents.

    Parameters
    ----------
    named:
        Pre-validated mapping from agent name → callable.
    default_agent:
        The callable for the built-in ``prompt`` keyword (or ``None`` if
        no default agent is configured).
    """

    def __init__(
        self,
        *,
        named: dict[str, AgentFn],
        default_agent: AgentFn | None,
    ) -> None:
        self._named = named
        self._default = default_agent

    @property
    def has_default_agent(self) -> bool:
        """True when a default agent backs the built-in ``prompt`` keyword."""
        return self._default is not None

    @property
    def agent_names(self) -> frozenset[str]:
        """Names of explicitly registered named agents."""
        return frozenset(self._named)

    def dispatch(self, name: str, request: AgentRequest) -> AgentResponse:
        """Dispatch a call to the appropriate agent callable.

        Resolution order:
        1. Named agents (exact match).
        2. Default agent: backs any *declared* name without a dedicated
           registration (scope guarantees only declared names reach here).

        Raises ``KeyError`` if the name is unknown and no default agent exists.

        ``AgentCallHostError`` raised by the callable is converted to
        ``AglRaise(ExceptionValue("AgentCallError", ...))`` so that the AgL
        interpreter can handle it as a catchable in-language exception.
        Transport failures are NOT eligible for ``on_parse_error`` retries
        (design §7.11): the ``AglRaise`` propagates directly to the
        interpreter's ``try/catch`` or the top-level ``WorkflowRuntime.run``
        dispatcher.
        """
        fn: AgentFn | None = self._named.get(name)
        if fn is None:
            # ``prompt`` and any declared named agent without a dedicated
            # registration both fall back to the default agent when configured.
            if self._default is not None:
                fn = self._default
            else:
                raise KeyError(f"No agent registered for name {name!r}")
        try:
            raw = fn(request)
        except AgentCallHostError as host_err:
            # Convert transport failure to a catchable AgL AgentCallError.
            _raise_agent_call_error(name, host_err)
        if isinstance(raw, str):
            return AgentResponse(content=raw)
        return raw


# ---------------------------------------------------------------------------
# AgentCallError construction helper (local import to avoid circular deps)
# ---------------------------------------------------------------------------


def _raise_agent_call_error(agent_name: str, err: AgentCallHostError) -> None:
    """Convert ``AgentCallHostError`` to ``AglRaise(ExceptionValue("AgentCallError", ...))``."""
    # Local imports keep runtime → eval dependency cycle-free at module level;
    # this mirrors the existing pattern in ``agm.agl.runtime.runtime``.
    from agm.agl.eval.exceptions import AglRaise
    from agm.agl.eval.values import JsonValue, TextValue
    from agm.agl.runtime.trace import new_trace_id

    metadata: dict[str, object] = {
        "exit_code": err.exit_code,
        "stderr_tail": err.stderr_tail,
        "elapsed": err.elapsed,
    }
    message = (
        f"Agent {agent_name!r} failed: {err.cause}"
        + (f" (exit {err.exit_code})" if err.exit_code is not None else "")
    )
    from agm.agl.eval.values import ExceptionValue

    exc_val = ExceptionValue(
        type_name="AgentCallError",
        fields={
            "message": TextValue(message),
            "trace_id": TextValue(new_trace_id()),
            "agent": TextValue(agent_name),
            "cause": TextValue(err.cause),
            "metadata": JsonValue(metadata),
        },
    )
    raise AglRaise(exc_val)


# ---------------------------------------------------------------------------
# Runner-backed agent factory (M5a)
# ---------------------------------------------------------------------------


def runner_backed_agent_factory(
    *,
    default_runner_cmd: str,
    per_agent_cmds: dict[str, str],
    idle_timeout: float | None,
) -> AgentFn:
    """Return an ``AgentFn`` that dispatches calls to an external runner process.

    Parameters
    ----------
    default_runner_cmd:
        The shell command used for agents not listed in *per_agent_cmds*.
        Split via ``shlex`` into ``argv``.
    per_agent_cmds:
        Per-agent command overrides (``[exec.agents]`` map from config).
        Keys are agent names; values are shell command strings.
    idle_timeout:
        Idle timeout (seconds) passed to ``run_prepared_prompt_result``.
        ``None`` means no timeout.
    """

    def agent_fn(request: AgentRequest) -> AgentResponse:
        from agm.agent.runner import (
            cleanup_temp_files,
            prepare_rendered_prompt_run,
            run_prepared_prompt_result,
        )

        # 1. Resolve the runner command for this agent.
        runner_cmd = per_agent_cmds.get(request.agent, default_runner_cmd)

        # 2. Compose the message text: rendered prompt + format_instructions
        #    + corrective feedback on retry (§9.5, §7.8).
        message_parts: list[str] = [request.prompt]

        contract = request.output_contract
        if contract is not None and contract.format_instructions:
            message_parts.append(contract.format_instructions)

        if request.attempt >= 1:
            # §7.8 corrective feedback
            validation_lines: list[str] = []
            for ve in request.validation_errors:
                validation_lines.append(f"- {ve.message}")
            val_block = "\n".join(validation_lines) if validation_lines else "(none)"
            prev_output = request.previous_invalid_output or ""
            retry_feedback = (
                "Your previous response did not match the required output format.\n\n"
                f"Validation errors:\n{val_block}\n\n"
                f"Previous response:\n{prev_output}\n\n"
                "Return only valid JSON matching the schema."
            )
            message_parts.append(retry_feedback)

        full_message = "\n\n".join(message_parts)

        # 3. Write to temp file (verbatim — no env-var expansion, §9.5).
        temp_files: list[Path] = []
        try:
            prepared = prepare_rendered_prompt_run(
                full_message,
                runner=runner_cmd,
                temp_files=temp_files,
                env=dict(os.environ),
            )

            # 4. Run and collect structured result.
            run_result = run_prepared_prompt_result(prepared, idle_timeout=idle_timeout)
        finally:
            cleanup_temp_files(temp_files)

        # 5. Map failures to AgentCallHostError.
        if run_result.spawn_error is not None:
            raise AgentCallHostError(
                cause="spawn_failure",
                exit_code=run_result.returncode,
                stderr_tail=_stderr_tail(run_result.stderr),
                elapsed=run_result.elapsed,
            )
        if run_result.timed_out:
            raise AgentCallHostError(
                cause="timeout",
                exit_code=run_result.returncode,
                stderr_tail=_stderr_tail(run_result.stderr),
                elapsed=run_result.elapsed,
            )
        if run_result.returncode is not None and run_result.returncode != 0:
            raise AgentCallHostError(
                cause="nonzero_exit",
                exit_code=run_result.returncode,
                stderr_tail=_stderr_tail(run_result.stderr),
                elapsed=run_result.elapsed,
            )

        # 6. Exit 0 with empty stdout is a valid empty response (plan §9.5).
        return AgentResponse(content=run_result.stdout, metadata={"elapsed": run_result.elapsed})

    return agent_fn


def _stderr_tail(stderr: str, *, max_chars: int = 500) -> str:
    """Return the last *max_chars* characters of *stderr* (the most useful tail)."""
    return stderr[-max_chars:] if len(stderr) > max_chars else stderr
