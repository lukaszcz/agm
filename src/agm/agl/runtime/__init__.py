"""AgL host runtime services package (eval-free).

Public API
----------
- :class:`AgentDeclInfo` — static summary of one ``agent`` declaration.
- :class:`CallSiteInfo` — static summary of one agent-call/exec site.
- :class:`AgentRequest` — request object passed to host agent callables.
- :class:`AgentResponse` — response from a host agent callable.
- :class:`AgentRegistry` — registry of host agents.
- :class:`OutputCodec` — protocol for output codecs.
- :class:`TextCodec` — the built-in passthrough text codec.
- :class:`JsonCodec` — the built-in structured-output codec.
- :class:`OutputContract` — materialized per-call output contract.
- :func:`render_value` — option-driven value renderer for interpolation, print,
  casts, builtin ``render``, and REPL echo.
"""

from __future__ import annotations

from agm.agl.runtime.agents import AgentFn, AgentRegistry
from agm.agl.runtime.codec import JsonCodec, OutputCodec, ParseResult, TextCodec
from agm.agl.runtime.contract import OutputContract
from agm.agl.runtime.render import render_value
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.runtime.types import AgentDeclInfo, CallSiteInfo

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
    "TextCodec",
    "render_value",
]
