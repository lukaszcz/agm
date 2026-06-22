"""Confirming agent wrapper for the AgL REPL.

Agent calls in AgL are real, costly, side-effecting LLM/shell invocations.  The
REPL gates them: in ``"confirm"`` mode the user approves (or declines) every live
call; in ``"auto"`` mode calls fire immediately (matching ``agm exec``).

:class:`ConfirmingAgent` is the single chokepoint that implements this.  It wraps
an underlying :data:`~agm.agl.runtime.agents.AgentFn`, holds the shared mutable
:class:`~agm.agl.repl.agentmode.AgentMode`, and delegates the actual prompting to
an injected *confirm* callback so the wrapper stays UI-free and unit-testable with
a fake callback.

Cancellation — a declined confirmation or a Ctrl-C during a live call — raises
:class:`AgentCancelled`.  The registry (``AgentRegistry.dispatch``) only catches
``AgentCallHostError``, so :class:`AgentCancelled` propagates out of the wrapped
callable into the session, which stops the current entry while preserving effects
completed before cancellation.

Ctrl-C handling: the runner subprocess runs in its own session/process group
(``agm.agent.runner`` calls ``run_capture_result(..., isolate_process_group=True)``,
which sets ``start_new_session=True``), so a terminal ``SIGINT`` does NOT reach it
directly.  Instead, on ``KeyboardInterrupt`` the parent's stream-draining
(``_drain_process_streams`` in ``core/process.py``) catches it, kills the
subprocess's process group, and re-raises — so the ``KeyboardInterrupt`` propagates
up into this wrapper, which converts it to :class:`AgentCancelled`.  No bespoke
process-killing is needed in the REPL layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from agm.agl.repl.agentmode import AgentMode
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.request import AgentRequest, AgentResponse


# The injected confirmation callback: given the callee name and the rendered
# prompt, it returns the user's decision.
ConfirmDecision = Literal["yes", "no", "always"]
ConfirmCallback = "Callable[[str, str], ConfirmDecision]"


class AgentCancelled(Exception):
    """Raised when a confirmed agent call is declined or interrupted.

    ``callee``
        The agent name the cancelled call targeted.
    ``reason``
        ``"declined"`` (the user answered ``no`` at the confirmation prompt) or
        ``"interrupted"`` (Ctrl-C during a live call).
    """

    def __init__(self, callee: str, reason: str) -> None:
        super().__init__(f"Agent call to {callee!r} cancelled ({reason}).")
        self.callee = callee
        self.reason = reason


class ConfirmingAgent:
    """An ``AgentFn`` that gates the underlying agent behind a confirmation flow.

    Holds a shared :class:`AgentMode` and an injected *confirm* callback.  In
    ``"auto"`` mode the underlying agent is dispatched immediately.  In
    ``"confirm"`` mode the callback decides:

    - ``"yes"``    → dispatch this call;
    - ``"always"`` → flip the shared mode to ``"auto"`` then dispatch (and every
      later call dispatches without prompting);
    - ``"no"``     → raise :class:`AgentCancelled` with reason ``"declined"``.

    A ``KeyboardInterrupt`` raised by the underlying agent during a live call is
    converted to :class:`AgentCancelled` with reason ``"interrupted"``.
    """

    def __init__(
        self,
        underlying: "AgentFn",
        mode: "AgentMode",
        *,
        confirm: "Callable[[str, str], ConfirmDecision]",
    ) -> None:
        self._underlying = underlying
        self._mode = mode
        self._confirm = confirm

    def __call__(self, request: "AgentRequest") -> "AgentResponse | str":
        if self._mode.mode == "confirm":
            decision = self._confirm(request.agent, request.prompt)
            if decision == "no":
                raise AgentCancelled(request.agent, "declined")
            if decision == "always":
                self._mode.mode = "auto"
            # ``"yes"`` and ``"always"`` both fall through to dispatch.
        return self._dispatch(request)

    def _dispatch(self, request: "AgentRequest") -> "AgentResponse | str":
        """Dispatch to the underlying agent, mapping Ctrl-C to a cancellation."""
        try:
            return self._underlying(request)
        except KeyboardInterrupt as exc:
            raise AgentCancelled(request.agent, "interrupted") from exc
