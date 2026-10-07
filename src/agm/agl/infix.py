"""Infix operator fixity, chain grouping, and the rewrite that replaces raw chains.

The parser keeps an expression applying binary operators or ``not`` prefixes
as a flat :class:`~agm.agl.syntax.nodes.RawInfixChain` until each operator's
fixity is known. A builtin operator is no name: its fixity is fixed, so the
parser groups a chain of builtin operators at once. A user operator is a name
whose fixity is that of the declaration scope selects for it, so scope groups
such a chain while it resolves it, then replaces every chain of the module in
one rewrite (:func:`replace_infix_chains`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import Field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, TypeGuard, cast

from agm.agl.diagnostics import AglSyntaxError
from agm.agl.syntax.nodes import (
    BinaryOp,
    BinOp,
    Call,
    Expr,
    InfixAssoc,
    InfixDecl,
    Program,
    RawInfixChain,
    RawInfixOperand,
    RawInfixOperator,
    UnaryNot,
    VarRef,
)
from agm.agl.syntax.spans import span_covering

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

Fixity = tuple[int, InfixAssoc]
"""An operator's priority and associativity."""

BUILTIN_FIXITIES: Mapping[str, Fixity] = {
    "or": (10, InfixAssoc.LEFT),
    "and": (20, InfixAssoc.LEFT),
    "in": (30, InfixAssoc.LEFT),
    "==": (30, InfixAssoc.LEFT),
    "!=": (30, InfixAssoc.LEFT),
    "<": (30, InfixAssoc.LEFT),
    "<=": (30, InfixAssoc.LEFT),
    ">": (30, InfixAssoc.LEFT),
    ">=": (30, InfixAssoc.LEFT),
    "+": (40, InfixAssoc.LEFT),
    "-": (40, InfixAssoc.LEFT),
    "*": (50, InfixAssoc.LEFT),
    "/": (50, InfixAssoc.LEFT),
}
"""The builtin operators, which no declaration can redeclare."""

BUILTIN_OPS: Mapping[str, BinOp] = {op.symbol: op for op in BinOp}
"""Each builtin operator by its spelling."""
_NON_ASSOCIATIVE: frozenset[str] = frozenset({"in", "==", "!=", "<", "<=", ">", ">="})
_DEFAULT_PRIORITY = 40
_NOT_PRIORITY = 25


def groups_at_parse(
    operands: Sequence[RawInfixOperand], operators: Sequence[RawInfixOperator]
) -> bool:
    """Whether the parser groups this chain: every operator is builtin and every operand grouped."""
    return all(operator.name in BUILTIN_FIXITIES for operator in operators) and not any(
        isinstance(operand.expr, RawInfixChain) for operand in operands
    )


def declared_priority(decl: InfixDecl, base: Callable[[str], Fixity]) -> int:
    """The priority *decl* declares; *base* gives the fixity its ``at prio`` operator names."""
    if decl.priority is not None:
        return decl.priority
    if decl.priority_base is not None:
        return base(decl.priority_base)[0] + decl.priority_delta
    return _DEFAULT_PRIORITY


def reject_builtin_redeclaration(decl: InfixDecl) -> None:
    """Reject *decl* declaring the fixity of a builtin operator."""
    if decl.name in BUILTIN_FIXITIES:
        raise AglSyntaxError(
            f"Cannot redeclare built-in operator '{decl.name}' as a user infix operator.",
            span=decl.span,
        )


def group_infix(
    operands: Sequence[RawInfixOperand],
    operators: Sequence[RawInfixOperator],
    fixities: Sequence[Fixity],
) -> Expr:
    """Group one chain whose ``i``-th operator has ``fixities[i]``.

    Operands are taken as written: a nested chain stays in place for
    :func:`replace_infix_chains`. A builtin operator applies as a
    :class:`BinaryOp`, any other as a call of the operator's name. Operators
    at one priority must agree on associativity, and a comparison grouped here
    takes no comparison grouped here as an operand.
    """
    associativity: dict[int, InfixAssoc] = {}
    for operator, (priority, assoc) in zip(operators, fixities, strict=True):
        if associativity.setdefault(priority, assoc) is not assoc:
            raise AglSyntaxError(
                "Operators at the same priority cannot mix left and right associativity.",
                span=operator.span,
            )
    prefix_nots = [list(operand.prefix_nots) for operand in operands]
    comparisons: set[int] = set()

    def prefixed(index: int) -> tuple[Expr, int]:
        if prefix_nots[index]:
            prefix = prefix_nots[index].pop(0)
            operand, following = climb(_NOT_PRIORITY, index)
            return UnaryNot(operand=operand, span=prefix.span, node_id=prefix.node_id), following
        return cast(Expr, operands[index].expr), index + 1

    def climb(minimum: int, index: int) -> tuple[Expr, int]:
        left, following = prefixed(index)
        while following - 1 < len(operators):
            operator = operators[following - 1]
            priority, assoc = fixities[following - 1]
            if priority < minimum:
                break
            right, following = climb(
                priority + 1 if assoc is InfixAssoc.LEFT else priority, following
            )
            left = apply(left, operator, right)
        return left, following

    def apply(left: Expr, operator: RawInfixOperator, right: Expr) -> Expr:
        span = span_covering(left.span, right.span)
        builtin = BUILTIN_OPS.get(operator.name)
        if builtin is None:
            callee = VarRef(name=operator.name, span=operator.span, node_id=operator.callee_node_id)
            return Call(
                callee=callee,
                args=(left, right),
                named_args=(),
                span=span,
                node_id=operator.node_id,
            )
        if operator.name in _NON_ASSOCIATIVE:
            if _grouped_comparison(left, comparisons) or _grouped_comparison(right, comparisons):
                raise AglSyntaxError(
                    "Comparisons are non-associative; parenthesize explicitly, "
                    "e.g. `(x == y) == z`.",
                    span=operator.span,
                )
            comparisons.add(operator.node_id)
        return BinaryOp(op=builtin, left=left, right=right, span=span, node_id=operator.node_id)

    grouped, _following = climb(0, 0)
    return grouped


def _grouped_comparison(expr: Expr, comparisons: set[int]) -> bool:
    return isinstance(expr, BinaryOp) and expr.node_id in comparisons


def replace_infix_chains(program: Program, grouped: Mapping[int, Expr]) -> Program:
    """Replace every chain of *program* by its grouping in *grouped*, keyed by chain id.

    A grouping holds its operands as written, so the chains nested in them
    are replaced too; a subtree holding no chain is kept as it is. A chain
    scope does not resolve -- one in an attribute argument other than a
    program's ``@config`` value -- has no grouping and stays as written, for
    attribute recognition to reject as no constant.
    """

    def rewrite(node: object) -> object:
        if isinstance(node, RawInfixChain):
            grouping = grouped.get(node.node_id)
            return node if grouping is None else rewrite(grouping)
        if isinstance(node, tuple):
            written = cast("tuple[object, ...]", node)
            items = tuple(rewrite(item) for item in written)
            return node if all(new is old for new, old in zip(items, written)) else items
        if not _is_dataclass_instance(node):
            return node
        changes: dict[str, object] = {}
        for spec in cast(tuple[Field[object], ...], fields(node)):
            old = cast(object, getattr(node, spec.name))
            new = rewrite(old)
            if new is not old:
                changes[spec.name] = new
        return replace(node, **changes) if changes else node

    return cast(Program, rewrite(program))


def _is_dataclass_instance(node: object) -> TypeGuard[DataclassInstance]:
    return is_dataclass(type(node))
