"""Which of a module's own static bindings an expression reads.

One module's ``let``/``var`` roots indexed by declaration, and the reverse
question answered over scope's resolution table rather than over spelling: a
reference reads a binding of this module exactly when ordinary name resolution
says it does. Constant classification and initializer ordering both ask it, so
they cannot disagree about which binding a name names.
"""

from __future__ import annotations

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.syntax.nodes import (
    Expr,
    LetDecl,
    VarDecl,
    VarRef,
    static_binding_node_id,
    static_items,
)
from agm.agl.syntax.visitor import walk

__all__ = ["ModuleBindingReferences"]


class ModuleBindingReferences:
    """One module's static ``let``/``var`` bindings, and who reads them."""

    def __init__(self, resolved: ModuleResolution, module_id: ModuleId) -> None:
        self._resolved = resolved
        self._module_id = module_id
        self.declarations: dict[int, LetDecl | VarDecl] = {
            static_binding_node_id(item): item
            for item in static_items(resolved.program.body.items)
            if isinstance(item, (LetDecl, VarDecl))
        }

    def declaration_for(self, node_id: int) -> LetDecl | VarDecl | None:
        """Return the static binding of this module the reference *node_id* names."""
        reference = self._resolved.resolution.get(node_id)
        if reference is None or reference.module_id != self._module_id:
            return None
        return self.declarations.get(reference.decl_node_id)

    def dependencies(self, expr: Expr) -> frozenset[int]:
        """Return the declarations of this module that *expr* reads."""
        found: set[int] = set()

        def collect(node: object) -> None:
            if isinstance(node, VarRef):
                declaration = self.declaration_for(node.node_id)
                if declaration is not None:
                    found.add(static_binding_node_id(declaration))

        walk(expr, collect)
        return frozenset(found)
