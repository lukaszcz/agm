"""Which of a module's static bindings a constant expression may name.

A constant expression may name a constant of its own module, so deciding
whether one reference is constant means deciding whether the binding behind it
has a constant initializer -- which may itself name a third binding. This walks
that chain on demand over :class:`ModuleBindingReferences`, so a reference
agrees with ordinary name resolution wherever it is written.

The chain is a fixpoint with one interesting failure: a binding defined in
terms of itself never becomes constant, and reporting that as "not a constant
expression" would name the symptom rather than the cause, so a cycle raises its
own diagnostic when the chain closes.
"""

from __future__ import annotations

from collections.abc import Callable

from agm.agl.modules.ids import ModuleId
from agm.agl.scope import ModuleResolution
from agm.agl.scope.bindings import ModuleBindingReferences
from agm.agl.syntax.constants import is_constant_expression
from agm.agl.syntax.nodes import Expr, static_binding_node_id
from agm.agl.typecheck.env import AglTypeError

__all__ = ["ModuleConstantBindings"]


class ModuleConstantBindings:
    """One module's static bindings, classified as constant or not.

    Built per module and queried per reference. ``is_constructor`` and
    ``is_constant_builtin`` are the caller's own classifications, so the
    constant rule a binding's initializer answers to is exactly the one the
    referring position answers to.
    """

    def __init__(
        self,
        resolved: ModuleResolution,
        module_id: ModuleId,
        *,
        is_constructor: Callable[[int], bool],
        is_constant_builtin: Callable[[int], bool] = lambda _node_id: False,
    ) -> None:
        self._references = ModuleBindingReferences(resolved, module_id)
        self._is_constructor = is_constructor
        self._is_constant_builtin = is_constant_builtin
        self._constant: dict[int, bool] = {}
        self._folding: list[int] = []

    def is_constant_reference(self, node_id: int) -> bool:
        """Whether the reference *node_id* names a constant of this module."""
        declaration = self._references.declaration_for(node_id)
        if declaration is None:
            return False
        return self._is_constant_binding(static_binding_node_id(declaration))

    def is_constant(self, expr: Expr) -> bool:
        """Whether *expr* is a constant expression in this module."""
        return is_constant_expression(
            expr,
            is_constructor=self._is_constructor,
            is_constant_builtin=self._is_constant_builtin,
            is_module_constant=self.is_constant_reference,
        )

    def dependencies(self, expr: Expr) -> frozenset[int]:
        """Return the declarations of this module that *expr* reads."""
        return self._references.dependencies(expr)

    def _is_constant_binding(self, declaration_node_id: int) -> bool:
        cached = self._constant.get(declaration_node_id)
        if cached is not None:
            return cached
        declaration = self._references.declarations[declaration_node_id]
        if declaration_node_id in self._folding:
            raise AglTypeError(
                f"Constant {declaration.name!r} is defined in terms of itself.",
                span=declaration.span,
            )
        self._folding.append(declaration_node_id)
        try:
            constant = self.is_constant(declaration.value)
        finally:
            self._folding.pop()
        self._constant[declaration_node_id] = constant
        return constant
