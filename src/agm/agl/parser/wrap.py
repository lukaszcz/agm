"""Pure AST transform for inline sources without an explicit program entry."""

from __future__ import annotations

from dataclasses import replace

import agm.agl.syntax as syntax
from agm.agl.attributes import PARAM_ATTRIBUTE
from agm.agl.syntax.nodes import Item, Program, static_function_items
from agm.agl.syntax.visitor import SyntaxNode, walk

_ROOT_DECLARATIONS = (
    syntax.FuncDef,
    syntax.RecordDef,
    syntax.EnumDef,
    syntax.ExceptionDef,
    syntax.TypeAlias,
    syntax.BuiltinVarDecl,
    syntax.InfixDecl,
    syntax.ImportDecl,
    syntax.ExportDecl,
    syntax.UseDecl,
    syntax.ScopeRegion,
)


def _binding_names(binding: syntax.LetDecl | syntax.VarDecl) -> frozenset[str]:
    """The name *binding* introduces."""
    return frozenset({binding.name})


def _pattern_names(pattern: syntax.Pattern) -> frozenset[str]:
    """The names *pattern* binds."""
    names: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, (syntax.VarPattern, syntax.AsPattern)):
            names.add(node.name)

    walk(pattern, visit)
    return frozenset(names)


def _shield(scope: SyntaxNode, names: frozenset[str], shielded: set[int]) -> None:
    """Record the bare references to *names* inside *scope*, which a binder of theirs encloses."""

    def visit(node: object) -> None:
        if (
            isinstance(node, (syntax.VarRef, syntax.NameTarget))
            and node.qualifier is None
            and node.name in names
        ):
            shielded.add(node.node_id)

    walk(scope, visit)


def _shield_binders(node: object, shielded: set[int]) -> None:
    """Record the bare references *node*'s own lexical binders enclose.

    A binder encloses only what it is visible to: a parameter the body (not
    the defaults), a loop variable the loop body, a pattern or ``catch``
    binding its branch, a block's ``let``/``var`` the items after it. A region
    binding declares a scope path, not a lexical binder.
    """
    if isinstance(node, (syntax.FuncDef, syntax.Lambda)) and node.body is not None:
        _shield(node.body, frozenset(param.name for param in node.params), shielded)
    elif isinstance(node, syntax.Loop) and node.for_var is not None:
        _shield(node.body, frozenset({node.for_var}), shielded)
    elif isinstance(node, syntax.CaseBranch):
        _shield(node.body, _pattern_names(node.pattern), shielded)
    elif isinstance(node, syntax.CatchClause) and node.binding is not None:
        _shield(node.body, frozenset({node.binding}), shielded)
    elif isinstance(node, syntax.Block):
        bound: frozenset[str] = frozenset()
        for item in node.items:
            _shield(item, bound, shielded)
            if isinstance(item, (syntax.LetDecl, syntax.VarDecl)):
                bound |= {item.name}


def _free_names(item: Item) -> frozenset[str]:
    """Names *item* may read from the module root.

    A read is a ``VarRef`` or an assignment target anywhere in the subtree,
    matched by member name so a qualified spelling matches its binding. Only a
    bare read that a lexical binder of its name encloses is dropped: it can
    never reach the root.
    """
    references: dict[int, str] = {}
    shielded: set[int] = set()

    def visit(node: object) -> None:
        if isinstance(node, (syntax.VarRef, syntax.NameTarget)):
            references[node.node_id] = node.name
        _shield_binders(node, shielded)

    walk(item, visit)
    return frozenset(name for node_id, name in references.items() if node_id not in shielded)


def _root_retained(items: tuple[Item, ...]) -> frozenset[int]:
    """Indices of the items that must stay at the program root.

    Declarations, path-bearing bindings, and ``@param`` bindings are retained
    outright — each is valid only at a module root. An unscoped ordinary
    ``let``/``var`` is retained only when a retained item reads its name,
    computed to a fixpoint so a retained binding's own initializer retains
    what it reads in turn.
    Constancy is not decided here: the checker rejects a retained binding whose
    initializer is not a constant expression, which the scope pass alone cannot
    determine.
    """
    retained = {
        index
        for index, item in enumerate(items)
        if isinstance(item, _ROOT_DECLARATIONS)
        or (
            isinstance(item, (syntax.LetDecl, syntax.VarDecl))
            and (
                bool(item.scope_path)
                or any(attr.name == PARAM_ATTRIBUTE for attr in item.attributes)
            )
        )
    }
    needed: set[str] = set()
    for index in retained:
        needed |= _free_names(items[index])
    candidates = {
        index: item
        for index, item in enumerate(items)
        if index not in retained and isinstance(item, (syntax.LetDecl, syntax.VarDecl))
    }
    growing = True
    while growing:
        growing = False
        for index, item in list(candidates.items()):
            if _binding_names(item) & needed:
                del candidates[index]
                retained.add(index)
                needed |= _free_names(item)
                growing = True
    return frozenset(retained)


def wrap_inline_program(program: Program, *, next_node_id: int) -> tuple[Program, int]:
    """Wrap root non-declarations in a host-only synthetic program entry.

    The transform is syntactic: root declarations, scope regions, path-bearing
    and ``@param`` bindings, and every binding a root item reads (see
    :func:`_root_retained`) stay at the root, while all other items move into
    ``main`` in source order.
    Retaining the bindings root declarations read keeps those declarations'
    normal textual visibility.

    If *program* already contains a ``program def`` at any scope path, it is
    returned unchanged with its supplied node-id seed. Otherwise the returned
    seed follows the three synthesized nodes, so callers can safely use it for
    subsequently parsed modules. Downstream passes use ``is_synthetic`` to keep
    the entry's node-id-based host lowering without introducing a source name.
    """
    if any(function.is_program for function in static_function_items(program.body.items)):
        return program, next_node_id

    items = program.body.items
    retained = _root_retained(items)
    root_items: list[syntax.Item] = []
    main_items: list[syntax.Item] = []
    has_executable_item = False
    for index, item in enumerate(items):
        is_module_header = isinstance(item, (syntax.ImportDecl, syntax.ExportDecl))
        if index in retained and not (is_module_header and has_executable_item):
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
