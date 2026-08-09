"""Public façade for the AgL agent workflow DSL.

The façade exports the pipeline API lazily so lower AGM domains can depend on
AgL leaves such as module identifiers without importing the execution stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.diagnostics import (
        AglError,
        Diagnostic,
        RelatedDiagnostic,
        SourceSpan,
        format_diagnostic,
    )
    from agm.agl.pipeline import PipelineDriver, PreparedProgram, RunError, RunResult
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.types import CallSiteInfo

__all__ = [
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


def __getattr__(name: str) -> object:
    """Load one public export only when a caller requests it."""
    if name == "AgentFn":
        from agm.agl.runtime.agents import AgentFn

        return AgentFn
    if name == "AglError":
        from agm.agl.diagnostics import AglError

        return AglError
    if name == "CallSiteInfo":
        from agm.agl.runtime.types import CallSiteInfo

        return CallSiteInfo
    if name == "Diagnostic":
        from agm.agl.diagnostics import Diagnostic

        return Diagnostic
    if name == "format_diagnostic":
        from agm.agl.diagnostics import format_diagnostic

        return format_diagnostic
    if name == "PipelineDriver":
        from agm.agl.pipeline import PipelineDriver

        return PipelineDriver
    if name == "RelatedDiagnostic":
        from agm.agl.diagnostics import RelatedDiagnostic

        return RelatedDiagnostic
    if name == "PreparedProgram":
        from agm.agl.pipeline import PreparedProgram

        return PreparedProgram
    if name == "RunError":
        from agm.agl.pipeline import RunError

        return RunError
    if name == "RunResult":
        from agm.agl.pipeline import RunResult

        return RunResult
    if name == "SourceSpan":
        from agm.agl.diagnostics import SourceSpan

        return SourceSpan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
