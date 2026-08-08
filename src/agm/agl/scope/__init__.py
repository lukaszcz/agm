"""AgL scope/name-resolution pass.

Public API
----------
- :func:`resolve_program` — whole-program static name-resolution pass over a
  :class:`~agm.agl.modules.loader.ModuleGraph`: one ``ModuleResolution`` per
  module.
- :class:`ModuleResolution` — frozen dataclass carrying one module's
  ``Program`` plus side tables keyed by ``node_id``.
- :class:`BindingRef` — resolved reference to a scope binding.
- :class:`BuiltinKind` — enum classifying contextual built-in Call nodes.
- :class:`AglScopeError` — fatal scope error (span-aware ``AglError``
  subclass).
"""

from __future__ import annotations

from agm.agl.scope.program import ResolvedModule, ResolvedProgram, resolve_program
from agm.agl.scope.symbols import (
    AglScopeError,
    BindingRef,
    BuiltinKind,
    ModuleResolution,
    ScopeNode,
)

__all__ = [
    "AglScopeError",
    "BindingRef",
    "BuiltinKind",
    "ModuleResolution",
    "ResolvedModule",
    "ResolvedProgram",
    "ScopeNode",
    "resolve_program",
]
