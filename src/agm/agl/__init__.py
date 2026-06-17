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
            print(f"line {d.line}: {d.message}")
"""

from __future__ import annotations

from agm.agl.diagnostics import AglError, Diagnostic, SourceSpan
from agm.agl.runtime.agents import AgentFn
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
    "AglError",
    "CallSiteInfo",
    "Diagnostic",
    "PreparedProgram",
    "RunError",
    "RunResult",
    "SourceSpan",
    "WorkflowRuntime",
]
