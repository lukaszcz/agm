"""Public façade for the AgL agent workflow DSL.

The façade exports the pipeline API lazily so lower AGM domains can depend on
AgL leaves such as module identifiers without importing the execution stack.
"""

from __future__ import annotations

from importlib import import_module
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

# name -> (module path, attribute name). Keeps every export a one-line entry
# and the lazy-loading mechanics (below) independent of the export list, so
# importing ``agm.agl`` itself never pulls in the execution stack.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentFn": ("agm.agl.runtime.agents", "AgentFn"),
    "AglError": ("agm.agl.diagnostics", "AglError"),
    "CallSiteInfo": ("agm.agl.runtime.types", "CallSiteInfo"),
    "Diagnostic": ("agm.agl.diagnostics", "Diagnostic"),
    "format_diagnostic": ("agm.agl.diagnostics", "format_diagnostic"),
    "PipelineDriver": ("agm.agl.pipeline", "PipelineDriver"),
    "RelatedDiagnostic": ("agm.agl.diagnostics", "RelatedDiagnostic"),
    "PreparedProgram": ("agm.agl.pipeline", "PreparedProgram"),
    "RunError": ("agm.agl.pipeline", "RunError"),
    "RunResult": ("agm.agl.pipeline", "RunResult"),
    "SourceSpan": ("agm.agl.diagnostics", "SourceSpan"),
}


def __getattr__(name: str) -> object:
    """Load one public export only when a caller requests it."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attribute = target
    module = import_module(module_path)
    value: object = getattr(module, attribute)
    return value
