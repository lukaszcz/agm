"""Pure AST transform for inline sources without an explicit program entry."""

from __future__ import annotations

from dataclasses import replace

import agm.agl.syntax as syntax
from agm.agl.syntax.constants import is_constant_expression
from agm.agl.syntax.nodes import Program, static_function_items


def wrap_inline_program(program: Program, *, next_node_id: int) -> tuple[Program, int]:
    """Wrap root non-declarations in a host-only synthetic program entry.

    The transform is syntactic: root declarations, scope regions, path-bearing
    bindings, and constant bindings preceding a function stay at the root,
    while every other item moves into ``main`` in source order. Keeping those
    earlier bindings lets root functions retain normal textual visibility.

    If *program* already contains a ``program def`` at any scope path, it is
    returned unchanged with its supplied node-id seed. Otherwise the returned
    seed follows the three synthesized nodes, so callers can safely use it for
    subsequently parsed modules. Downstream passes use ``is_synthetic`` to keep
    the entry's node-id-based host lowering without introducing a source name.
    """
    if any(function.is_program for function in static_function_items(program.body.items)):
        return program, next_node_id

    items = program.body.items
    root_items: list[syntax.Item] = []
    main_items: list[syntax.Item] = []
    has_executable_item = False
    for index, item in enumerate(items):
        is_module_header = isinstance(item, (syntax.ImportDecl, syntax.ExportDecl))
        if is_module_header and has_executable_item:
            main_items.append(item)
            continue
        is_scoped_binding = isinstance(item, (syntax.LetDecl, syntax.VarDecl)) and bool(
            item.scope_path
        )
        is_earlier_constant_binding = (
            isinstance(item, (syntax.LetDecl, syntax.VarDecl))
            and not item.scope_path
            and any(isinstance(later, syntax.FuncDef) for later in items[index + 1 :])
            and is_constant_expression(item.value, is_constructor=lambda _node_id: False)
        )
        if (
            is_scoped_binding
            or is_earlier_constant_binding
            or isinstance(
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
            )
        ):
            root_items.append(item)
        else:
            main_items.append(item)
            has_executable_item = True

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
        is_synthetic=True,
    )
    body = replace(program.body, items=(*root_items, main))
    return replace(program, body=body), next_node_id + 3
