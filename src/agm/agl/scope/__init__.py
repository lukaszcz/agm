"""AgL scope/name-resolution pass.

Public API
----------
- :func:`resolve_module` — per-module static name-resolution pass: ``Program →
  ModuleResolution``.
- :class:`ModuleResolution` — frozen dataclass carrying the ``Program`` plus
  side tables keyed by ``node_id``.
- :class:`BindingRef` — resolved reference to a scope binding.
- :class:`BuiltinKind` — enum classifying contextual built-in Call nodes.
- :class:`AglScopeError` — fatal scope error (span-aware ``AglError``
  subclass).
"""

from __future__ import annotations

from agm.agl.scope.program import ResolvedModule, ResolvedProgram, resolve_program
from agm.agl.scope.resolver import resolve_module
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
    "resolve_module",
    "resolve_program",
]
