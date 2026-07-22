"""Shared function-header models and resolution helpers.

Candidate return inference will extend this seam without exposing provisional
state in semantic function signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.persistent import PersistentDict
from agm.agl.semantics.types import FunctionType, Type, contains_inference_var
from agm.agl.syntax.nodes import FuncDef, Param, ParamKind, Program, VarRef
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.inference import ConstraintRole, InferenceEngine

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.scope.symbols import ModuleResolution
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.env import AglTypeError, FunctionSignature, ParamSpec, TypeEnvironment


@dataclass(slots=True)
class CandidateSession:
    """Disposable state shared while discovering one candidate signature.

    Candidate checkers use this session's engine and retain their expression
    side tables only for the duration of body traversal.  The final checker
    creates all published artifacts after the candidate is concrete.
    """

    engine: InferenceEngine
    provisional_declaration_ids: frozenset[int]
    evidence: list[SourceSpan] = field(default_factory=list)
    binding_snapshot: PersistentDict[int, Type] | None = None


@dataclass(frozen=True, slots=True)
class CandidateModule:
    """One module participating in disposable candidate discovery."""

    resolved: "ModuleResolution"
    env: TypeEnvironment
    capabilities: "HostCapabilities"
    module_id: ModuleId


@dataclass(frozen=True, slots=True)
class ModuleCandidateComponent:
    """A module component whose candidate signatures are coordinated together.

    The standalone checker creates only synthetic singleton components.
    Import-SCC inference can extend this boundary without changing that path.
    """

    modules: tuple[CandidateModule, ...]

    @classmethod
    def singleton(
        cls,
        resolved: "ModuleResolution",
        env: TypeEnvironment,
        capabilities: "HostCapabilities",
        module_id: ModuleId,
    ) -> "ModuleCandidateComponent":
        """Build the standalone checker's synthetic one-module component."""
        return cls((CandidateModule(resolved, env, capabilities, module_id),))


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


def _has_direct_self_reference(node: FuncDef, resolved: "ModuleResolution") -> bool:
    """Return whether a function body references its own resolved declaration."""
    assert node.body is not None
    found = False

    def visit(item: object) -> None:
        nonlocal found
        if isinstance(item, VarRef):
            reference = resolved.resolution.get(item.node_id)
            found = found or (reference is not None and reference.decl_node_id == node.node_id)

    walk(node.body, visit)
    return found


def infer_module_component_candidates(component: ModuleCandidateComponent) -> None:
    """Discover candidates for one module component before authoritative checking.

    The current caller supplies a synthetic singleton component.  This
    coordinator is deliberately the extension point for later import/SCC work;
    direct self-recursion remains the only candidate shape discovered here.
    """
    for module in component.modules:
        _infer_direct_recursive_candidates(module)


def _infer_direct_recursive_candidates(module: CandidateModule) -> None:
    """Close a module's directly recursive unannotated monomorphic functions."""
    from agm.agl.typecheck.checker import _Checker

    program = module.resolved.program
    assert isinstance(program, Program)
    for item in program.body.items:
        if (
            not isinstance(item, FuncDef)
            or item.return_type is not None
            or item.is_builtin
            or item.is_extern
            or item.type_params
            or not _has_direct_self_reference(item, module.resolved)
        ):
            continue
        checker = _Checker(
            env=module.env,
            resolved=module.resolved,
            capabilities=module.capabilities,
            module_id=module.module_id,
        )
        checker._validate_funcdef_header(item)
        engine = InferenceEngine()
        result = engine.fresh(f"{item.name} result")
        signature, function_type = resolve_function_header(module.env, item, result_type=result)
        module.env.register_function_signature(item.name, signature)
        module.env.register_function_signature_by_node_id(item.node_id, signature)
        module.env.set_binding_type(item.node_id, function_type)
        session = CandidateSession(engine, frozenset({item.node_id}))
        session.binding_snapshot = module.env.snapshot_binding_types()
        try:
            candidate_type = checker.check_candidate_funcdef_body(item, signature, session)
            engine.unify(
                result,
                candidate_type,
                engine.origin(
                    item.span,
                    role=ConstraintRole.EXPECTED_RESULT,
                    subject=f"return type of function '{item.name}'",
                ),
            )
        finally:
            assert session.binding_snapshot is not None
            module.env.restore_binding_types(session.binding_snapshot)
        concrete_result = engine.zonk(result)
        if not engine.is_solved(result) or contains_inference_var(concrete_result):
            raise AglTypeError(
                f"Cannot infer return type of function '{item.name}': insufficient concrete "
                "return evidence. Add a return type annotation.",
                span=item.span,
            )
        concrete_signature = FunctionSignature(
            params=signature.params,
            result=concrete_result,
            type_params=signature.type_params,
        )
        module.env.register_function_signature(item.name, concrete_signature)
        module.env.register_function_signature_by_node_id(item.node_id, concrete_signature)
        module.env.set_binding_type(
            item.node_id,
            FunctionType(
                params=tuple(param.type for param in signature.params), result=concrete_result
            ),
        )


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
