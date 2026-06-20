"""Public façade for the AgL agent workflow DSL.

Usage::

    from agm.agl import WorkflowRuntime

    runtime = WorkflowRuntime(
        default_loop_limit=5,
        default_strict_json=False,
        default_agent=my_agent_fn,
    )
    runtime.register_agent("reviewer", reviewer_fn)
    result = runtime.run(source_text, param_values={"spec": "..."})

    if result.ok:
        ...
    else:
        for d in result.diagnostics:
            print(format_diagnostic(d))
"""

from __future__ import annotations

from agm.agl.diagnostics import AglError, Diagnostic, SourceSpan, format_diagnostic
from agm.agl.runtime.agents import AgentFn
from agm.agl.runtime.runtime import (
    AgentDeclInfo,
    CallSiteInfo,
    PreparedGraph,
    PreparedProgram,
    RunError,
    RunResult,
    WorkflowRuntime,
)

__all__ = [
    "AgentDeclInfo",
    "AgentFn",
    "AglError",
    "CallSiteInfo",
    "Diagnostic",
    "format_diagnostic",
    "PreparedGraph",
    "PreparedProgram",
    "RunError",
    "RunResult",
    "SourceSpan",
    "WorkflowRuntime",
]
