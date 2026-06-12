"""Runtime scope for the AgL evaluator.

``Scope`` is a linked chain of frames (parent → child) opened at each
scope-introducing construct.  ``Binding`` records the name, value, mutability
flag, and declaration span for one variable in that frame.

The root scope is the only one that persists into ``RunResult.bindings``;
nested scopes are discarded when the construct that opened them exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agm.agl.eval.values import Value
from agm.agl.syntax.spans import SourceSpan


@dataclass(slots=True)
class Binding:
    """A single named binding in the current scope frame."""

    name: str
    value: Value
    mutable: bool
    decl_span: SourceSpan


@dataclass(slots=True)
class Scope:
    """One frame in the runtime scope chain.

    ``parent`` is the enclosing scope (``None`` at the root).
    ``bindings`` maps names to ``Binding`` objects introduced in *this* frame
    only; lookup walks up the parent chain.
    """

    parent: Optional["Scope"]
    bindings: dict[str, Binding]

    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self.bindings = {}

    def define(self, name: str, value: Value, *, mutable: bool, decl_span: SourceSpan) -> None:
        """Introduce *name* into this frame."""
        self.bindings[name] = Binding(
            name=name, value=value, mutable=mutable, decl_span=decl_span
        )

    def lookup(self, name: str) -> Binding | None:
        """Find *name* by walking up the scope chain."""
        scope: Scope | None = self
        while scope is not None:
            b = scope.bindings.get(name)
            if b is not None:
                return b
            scope = scope.parent
        return None

    def set_value(self, name: str, value: Value) -> bool:
        """Update *name*'s value in the nearest frame that contains it.

        Returns ``True`` on success, ``False`` if *name* is not found.
        Does NOT enforce mutability — the static pass already handles that.
        """
        scope: Scope | None = self
        while scope is not None:
            if name in scope.bindings:
                scope.bindings[name].value = value
                return True
            scope = scope.parent
        return False

    def snapshot(self) -> dict[str, Value]:
        """Return a flat mapping of all names visible in this scope chain.

        For the root scope this gives the program's final state.  Names
        defined in inner scopes shadow outer ones (consistent with lookup).
        """
        result: dict[str, Value] = {}
        scope: Scope | None = self
        # Collect from outermost to innermost so inner shadows outer.
        frames: list[dict[str, Binding]] = []
        while scope is not None:
            frames.append(scope.bindings)
            scope = scope.parent
        for frame in reversed(frames):
            for name, binding in frame.items():
                result[name] = binding.value
        return result
