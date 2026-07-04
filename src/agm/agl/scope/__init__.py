"""AgL scope/name-resolution pass.

Public API
----------
- :func:`resolve` — full static name-resolution pass: ``Program →
  ResolvedProgram``.
- :class:`ResolvedProgram` — frozen dataclass carrying the ``Program`` plus
  side tables keyed by ``node_id``.
- :class:`BindingRef` — resolved reference to a scope binding.
- :class:`BuiltinKind` — enum classifying contextual built-in Call nodes.
- :class:`AglScopeError` — fatal scope error (span-aware ``AglError``
  subclass).
"""

from __future__ import annotations

from agm.agl.scope.resolver import resolve
from agm.agl.scope.symbols import (
    AglScopeError,
    BindingRef,
    BuiltinKind,
    ResolvedProgram,
    ScopeNode,
)

__all__ = [
    "AglScopeError",
    "BindingRef",
    "BuiltinKind",
    "ResolvedProgram",
    "ScopeNode",
    "resolve",
]
