"""Shared function-header models and resolution helpers.

Candidate return inference will extend this seam without exposing provisional
state in semantic function signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from agm.agl.semantics.types import FunctionType, Type
from agm.agl.syntax.nodes import FuncDef, Param, ParamKind
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.env import AglTypeError, FunctionSignature, ParamSpec, TypeEnvironment


class FunctionReturnSource(Enum):
    """The source of a program-level function result type."""

    DECLARED = "declared"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class FunctionSignatureRecord:
    """Internal program-signature metadata keyed by a declaration node id.

    ``candidate_evidence`` is reserved for the spans that support a later
    candidate-inferred result and remains empty for declared results.
    """

    declaration_node_id: int
    name: str
    signature: FunctionSignature
    function_type: FunctionType
    is_builtin: bool
    is_extern: bool
    return_source: FunctionReturnSource
    candidate_evidence: tuple[SourceSpan, ...] = ()


def validate_required_after_defaulted(params: Sequence[Param]) -> None:
    """Reject a required positional-fillable parameter after a defaulted one."""
    seen_pos_default = False
    for param in params:
        is_pos_fillable = param.kind in (ParamKind.POSITIONAL_ONLY, ParamKind.STANDARD)
        if not is_pos_fillable:
            continue
        if param.default is not None:
            seen_pos_default = True
        elif seen_pos_default:
            raise AglTypeError(
                f"Parameter '{param.name}' has no default but follows a defaulted "
                "positional parameter. Required positional parameters must come "
                "before parameters with defaults.",
                span=param.span,
            )


def resolve_function_header(
    env: TypeEnvironment,
    node: FuncDef,
    *,
    result_type: TypeExpr | Type,
) -> tuple[FunctionSignature, FunctionType]:
    """Resolve one function's parameter scheme and declared or supplied result."""
    validate_required_after_defaulted(node.params)
    type_vars = frozenset(node.type_params)
    params = tuple(
        ParamSpec(
            name=param.name,
            type=env.resolve_type_expr(param.type_expr, span=param.span, type_vars=type_vars),
            kind=param.kind,
            has_default=param.default is not None,
        )
        for param in node.params
    )
    resolved_result = (
        env.resolve_type_expr(result_type, span=node.span, type_vars=type_vars)
        if isinstance(result_type, TypeExpr)
        else result_type
    )
    signature = FunctionSignature(
        params=params,
        result=resolved_result,
        type_params=node.type_params,
    )
    return signature, FunctionType(
        params=tuple(param.type for param in params), result=resolved_result
    )
