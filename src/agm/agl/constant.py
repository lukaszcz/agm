"""Typed parsing and static validation for AgL constant expressions.

A constant expression is built only from literal syntax, container literals, and
constructor applications. Type checking remains responsible for deciding which
references are constructors and for validating the expression's declared type.
"""

from __future__ import annotations

from collections.abc import Callable

from agm.agl.semantics.types import Type
from agm.agl.semantics.values import Value
from agm.agl.syntax import (
    ArrayLit,
    BoolLit,
    Call,
    DecimalLit,
    DictLit,
    Expr,
    IntLit,
    NullLit,
    StringLit,
    TypeApply,
    UnitLit,
    VarRef,
)

__all__ = ["ConstantExpressionError", "is_constant_expression", "parse_constant"]


class ConstantExpressionError(ValueError):
    """A source string cannot produce a constant value of its expected type."""


def parse_constant(source: str, expected_type: Type) -> Value:
    """Parse, typecheck, and evaluate one constant AgL expression.

    The expression is checked through the ordinary program pipeline, which
    applies :func:`is_constant_expression` using its resolved constructor
    metadata. The resulting runtime value is therefore exactly the value an
    AgL program would construct.

    Raises:
        ConstantExpressionError: If *source* is not one expression, fails to
            parse or typecheck, or is not constant. Every error includes the
            supplied source string so config and CLI callers can identify the
            failing input.
    """
    from agm.agl.modules.ids import ENTRY_ID
    from agm.agl.parser import AglSyntaxError, parse_program
    from agm.agl.pipeline import PipelineDriver
    from agm.agl.syntax import LetDecl

    try:
        program = parse_program(source)
    except AglSyntaxError as exc:
        message = f"Invalid AgL constant {source!r}: parse error: {exc}"
        raise ConstantExpressionError(message) from exc

    if len(program.body.items) != 1 or not isinstance(program.body.items[0], Expr):
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: expected exactly one expression."
        )

    driver = PipelineDriver()
    prepared = driver.prepare_program(
        f"let constant_value: {expected_type!r} = (\n{source}\n)\nconstant_value",
        default_stdlib=True,
    )
    discovery = driver.discover_params(prepared)
    if discovery.diagnostics:
        diagnostic = discovery.diagnostics[0]
        kind = "non-constant expression" if "is not defined" in diagnostic.message else "type error"
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: {kind}: {diagnostic.message}"
        )
    checked = discovery.checked
    compiled = discovery.compiled
    assert checked is not None
    assert compiled is not None

    entry_module = checked.modules[ENTRY_ID]
    constant_decl = next(
        item for item in entry_module.resolved.program.body.items if isinstance(item, LetDecl)
    )
    if not is_constant_expression(
        constant_decl.value,
        is_constructor=lambda node_id: entry_module.constructor_ref_for(node_id) is not None,
    ):
        raise ConstantExpressionError(
            f"Invalid AgL constant {source!r}: non-constant expression "
            "(constructors and literals only)."
        )

    result = driver.run_prepared(prepared, compiled=compiled)
    return result.bindings["constant_value"]


def is_constant_expression(expr: Expr, *, is_constructor: Callable[[int], bool]) -> bool:
    """Whether *expr* contains only literal construction.

    ``is_constructor`` is supplied by the checked frontend artifact, keeping
    this syntax-level predicate independent of scope and typecheck internals.
    """
    if isinstance(expr, (BoolLit, DecimalLit, IntLit, NullLit, StringLit, UnitLit)):
        return True
    if isinstance(expr, ArrayLit):
        return all(
            is_constant_expression(element, is_constructor=is_constructor)
            for element in expr.elements
        )
    if isinstance(expr, DictLit):
        return all(
            is_constant_expression(entry.value, is_constructor=is_constructor)
            for entry in expr.entries
        )
    if isinstance(expr, VarRef):
        return is_constructor(expr.node_id)
    if isinstance(expr, TypeApply):
        return is_constant_expression(expr.expr, is_constructor=is_constructor)
    if isinstance(expr, Call):
        return (
            is_constant_expression(expr.callee, is_constructor=is_constructor)
            and all(
                is_constant_expression(argument, is_constructor=is_constructor)
                for argument in expr.args
            )
            and all(
                is_constant_expression(argument.value, is_constructor=is_constructor)
                for argument in expr.named_args
            )
        )
    return False
