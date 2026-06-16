"""AgL scope/name-resolution pass (Component 4).

Public API
----------
- :func:`resolve` — full static name-resolution pass: ``Program →
  ResolvedProgram``.
- :class:`ResolvedProgram` — frozen dataclass carrying the ``Program`` plus
  side tables keyed by ``node_id``.
- :class:`BindingRef` — resolved reference to a scope binding.
- :class:`CallKind` — enum distinguishing ``agent``, ``default_agent``, and
  ``shell_exec`` calls.
- :class:`AglScopeError` — fatal scope error (span-aware ``AglError``
  subclass).

Note (v2 rewrite in progress)
------------------------------
The ``resolve`` import is deferred because ``resolver.py`` references AST nodes
that were removed/renamed by the S1a AST contract; eager import would crash at
module load until the resolver is rewritten.  To keep ``__all__`` honest during
this window, ``"resolve"`` is added to ``__all__`` only under ``TYPE_CHECKING``
(so ``from agm.agl.scope import *`` does not claim an export it cannot serve).

TODO(S2): rewrite resolver.py for the v2 AST, restore the eager
``from agm.agl.scope.resolver import resolve`` import, and move ``"resolve"``
back into the unconditional ``__all__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agm.agl.scope.symbols import AglScopeError, BindingRef, CallKind, ResolvedProgram, ScopeNode

__all__ = [
    "AglScopeError",
    "BindingRef",
    "CallKind",
    "ResolvedProgram",
    "ScopeNode",
]

if TYPE_CHECKING:
    from agm.agl.scope.resolver import resolve

    __all__ += ["resolve"]
