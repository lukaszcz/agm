"""Public façade for the AgL agent workflow DSL.

Usage::

    from agm.agl import PipelineDriver

    driver = PipelineDriver(
        default_strict_json=False,
        default_agent=my_agent_fn,
    )
    driver.register_agent("reviewer", reviewer_fn)
    result = driver.run(source_text, param_values={"spec": "..."})

    if result.ok:
        ...
    else:
        for d in result.diagnostics:
            print(format_diagnostic(d))
"""

from __future__ import annotations

from agm.agl.diagnostics import (
    AglError,
    Diagnostic,
    RelatedDiagnostic,
    SourceSpan,
    format_diagnostic,
)
from agm.agl.pipeline import (
    AgentDeclInfo,
    CallSiteInfo,
    PipelineDriver,
    PreparedProgram,
    RunError,
    RunResult,
)
from agm.agl.runtime.agents import AgentFn

__all__ = [
    "AgentDeclInfo",
    "AgentFn",
    "AglError",
    "CallSiteInfo",
    "Diagnostic",
    "format_diagnostic",
    "PipelineDriver",
    "RelatedDiagnostic",
    "PreparedProgram",
    "RunError",
    "RunResult",
    "SourceSpan",
]
