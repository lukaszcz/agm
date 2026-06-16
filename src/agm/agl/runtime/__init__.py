"""AgL host runtime package.

Public API
----------
- :class:`WorkflowRuntime` — the main host API façade.
- :class:`RunResult` — result of a ``WorkflowRuntime.run`` call.
- :class:`RunError` — structured uncaught AgL exception.
- :class:`CallSiteInfo` — static summary of one agent-call/exec site.
- :class:`AgentDeclInfo` — static summary of one ``agent`` declaration.
- :class:`AgentRequest` — request object passed to host agent callables.
- :class:`AgentResponse` — response from a host agent callable.
- :class:`AgentRegistry` — registry of host agents.
- :class:`OutputCodec` — protocol for output codecs.
- :class:`TextCodec` — the built-in passthrough text codec.
- :class:`JsonCodec` — the built-in structured-output codec (M2).
- :class:`OutputContract` — materialized per-call output contract.
"""

from __future__ import annotations

from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.codec import JsonCodec, OutputCodec, ParseResult, TextCodec
from agm.agl.runtime.contract import OutputContract
from agm.agl.runtime.render import render_for_console, render_for_prompt
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.runtime.runtime import (
    AgentDeclInfo,
    CallSiteInfo,
    PreparedProgram,
    RunError,
    RunResult,
    WorkflowRuntime,
)

__all__ = [
    "AgentDeclInfo",
    "AgentFn",
    "AgentRegistry",
    "AgentRequest",
    "AgentResponse",
    "CallSiteInfo",
    "JsonCodec",
    "OutputCodec",
    "OutputContract",
    "ParseResult",
    "PreparedProgram",
    "RunError",
    "RunResult",
    "TextCodec",
    "WorkflowRuntime",
    "render_for_console",
    "render_for_prompt",
]
