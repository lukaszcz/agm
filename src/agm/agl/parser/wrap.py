"""Pure AST transform for inline sources without an explicit program entry."""

from __future__ import annotations

from dataclasses import replace

import agm.agl.syntax as syntax
from agm.agl.syntax.nodes import Program, static_function_items


def wrap_inline_program(program: Program, *, next_node_id: int) -> tuple[Program, int]:
    """Wrap root non-declarations in a synthesized ``program def main``.

    The transform is intentionally syntactic: every root declaration and scope
    region stays at the root, while every other item moves into ``main`` in
    source order. It never inspects initializer constancy or resolves names.

    If *program* already contains a ``program def`` at any scope path, it is
    returned unchanged with its supplied node-id seed. Otherwise the returned
    seed follows the three synthesized nodes, so callers can safely use it for
    subsequently parsed modules.
    """
    if any(function.is_program for function in static_function_items(program.body.items)):
        return program, next_node_id

    root_items: list[syntax.Item] = []
    main_items: list[syntax.Item] = []
    for item in program.body.items:
        if isinstance(
            item,
            (
                syntax.FuncDef,
                syntax.RecordDef,
                syntax.EnumDef,
                syntax.ExceptionDef,
                syntax.TypeAlias,
                syntax.ParamDecl,
                syntax.BuiltinVarDecl,
                syntax.InfixDecl,
                syntax.ImportDecl,
                syntax.ExportDecl,
                syntax.OpenDecl,
                syntax.ScopeRegion,
            ),
        ):
            root_items.append(item)
        else:
            main_items.append(item)

    main_body = syntax.Block(
        items=tuple(main_items),
        span=program.body.span,
        node_id=next_node_id,
    )
    main = syntax.FuncDef(
        name="main",
        params=(),
        return_type=syntax.UnitT(span=program.span, node_id=next_node_id + 1),
        body=main_body,
        span=program.span,
        node_id=next_node_id + 2,
        is_program=True,
    )
    body = replace(program.body, items=(*root_items, main))
    return replace(program, body=body), next_node_id + 3
