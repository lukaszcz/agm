"""Tests for the confirming agent wrapper (``agm.agl.repl.agents``).

Drives :class:`ConfirmingAgent` with a fake underlying agent, a scripted
``confirm`` callback, and a real :class:`AgentMode`.  Asserts the confirm/auto
gating, the ``always`` mode flip, ``no`` cancellation, and Ctrl-C → cancellation
conversion — all without a real terminal or subprocess.
"""

from __future__ import annotations

import pytest

from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import AgentCancelled, ConfirmingAgent
from agm.agl.runtime.request import AgentRequest, AgentResponse


class RecordingAgent:
    """A fake ``AgentFn`` recording every dispatched request."""

    def __init__(self, reply: str = "ok") -> None:
        self.requests: list[AgentRequest] = []
        self._reply = reply

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        return AgentResponse(content=self._reply)


class InterruptingAgent:
    """A fake ``AgentFn`` that raises ``KeyboardInterrupt`` (simulated Ctrl-C)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
        raise KeyboardInterrupt


def _request(agent: str = "ask", prompt: str = "hi") -> AgentRequest:
    return AgentRequest(agent=agent, prompt=prompt)


class ScriptedConfirm:
    """A fake confirm callback returning scripted decisions and recording args."""

    def __init__(self, *decisions: str) -> None:
        self._decisions = list(decisions)
        self.seen: list[tuple[str, str]] = []
        self.calls = 0

    def __call__(self, callee: str, prompt: str) -> str:
        self.seen.append((callee, prompt))
        decision = self._decisions[min(self.calls, len(self._decisions) - 1)]
        self.calls += 1
        return decision


class TestConfirmMode:
    def test_yes_dispatches(self) -> None:
        underlying = RecordingAgent("reply")
        confirm = ScriptedConfirm("yes")
        wrapper = ConfirmingAgent(underlying, AgentMode(mode="confirm"), confirm=confirm)

        result = wrapper(_request(agent="writer", prompt="draft it"))

        assert isinstance(result, AgentResponse)
        assert result.content == "reply"
        assert len(underlying.requests) == 1
        # The callback was shown the callee and rendered prompt.
        assert confirm.seen == [("writer", "draft it")]

    def test_no_raises_cancelled_and_does_not_dispatch(self) -> None:
        underlying = RecordingAgent()
        wrapper = ConfirmingAgent(
            underlying, AgentMode(mode="confirm"), confirm=ScriptedConfirm("no")
        )

        with pytest.raises(AgentCancelled) as excinfo:
            wrapper(_request(agent="writer"))

        assert excinfo.value.callee == "writer"
        assert excinfo.value.reason == "declined"
        assert underlying.requests == []

    def test_always_flips_mode_and_skips_later_prompts(self) -> None:
        underlying = RecordingAgent()
        mode = AgentMode(mode="confirm")
        confirm = ScriptedConfirm("always", "no")  # 2nd would decline if asked
        wrapper = ConfirmingAgent(underlying, mode, confirm=confirm)

        wrapper(_request())
        assert mode.mode == "auto"
        assert len(underlying.requests) == 1

        # A subsequent call must NOT consult the confirm callback again.
        wrapper(_request())
        assert confirm.calls == 1  # only the first call prompted
        assert len(underlying.requests) == 2


class TestAutoMode:
    def test_auto_never_prompts(self) -> None:
        underlying = RecordingAgent()
        confirm = ScriptedConfirm("no")  # would decline if ever consulted
        wrapper = ConfirmingAgent(underlying, AgentMode(mode="auto"), confirm=confirm)

        wrapper(_request())
        wrapper(_request())

        assert confirm.calls == 0
        assert len(underlying.requests) == 2


class TestInterrupt:
    def test_keyboard_interrupt_becomes_cancelled(self) -> None:
        underlying = InterruptingAgent()
        wrapper = ConfirmingAgent(
            underlying, AgentMode(mode="auto"), confirm=ScriptedConfirm("yes")
        )

        with pytest.raises(AgentCancelled) as excinfo:
            wrapper(_request(agent="slow"))

        assert excinfo.value.callee == "slow"
        assert excinfo.value.reason == "interrupted"
        assert underlying.calls == 1

    def test_interrupt_after_confirm_yes_is_cancelled(self) -> None:
        underlying = InterruptingAgent()
        wrapper = ConfirmingAgent(
            underlying, AgentMode(mode="confirm"), confirm=ScriptedConfirm("yes")
        )

        with pytest.raises(AgentCancelled) as excinfo:
            wrapper(_request(agent="slow"))

        assert excinfo.value.reason == "interrupted"
