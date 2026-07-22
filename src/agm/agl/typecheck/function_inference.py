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
from agm.agl.semantics.types import FunctionType, InferenceVarType, Type, contains_inference_var
from agm.agl.syntax.nodes import FuncDef, Param, ParamKind, Program, VarRef
from agm.agl.syntax.visitor import walk
from agm.agl.typecheck.inference import ConstraintRole, InferenceEngine, InferenceError
from agm.util.graph import sccs

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


def _declaration_key(node: FuncDef) -> tuple[int, int]:
    """Return the stable source order for a top-level function declaration."""
    return (node.span.start_offset, node.node_id)


def _candidate_functions(module: CandidateModule) -> dict[int, FuncDef]:
    """Return unannotated monomorphic ordinary functions by declaration id."""
    program = module.resolved.program
    assert isinstance(program, Program)
    return {
        item.node_id: item
        for item in program.body.items
        if isinstance(item, FuncDef)
        and item.return_type is None
        and not item.is_builtin
        and not item.is_extern
        and not item.type_params
    }


def _function_key(functions: dict[int, FuncDef], declaration_id: int) -> tuple[int, int]:
    """Look up the stable source order for a declaration id."""
    return _declaration_key(functions[declaration_id])


def _function_dependencies(
    functions: dict[int, FuncDef], resolved: "ModuleResolution"
) -> dict[int, tuple[int, ...]]:
    """Collect body references to candidate functions by resolved declaration id."""
    dependencies: dict[int, tuple[int, ...]] = {}
    for declaration_id, node in functions.items():
        assert node.body is not None
        referenced: set[int] = set()

        def visit(item: object) -> None:
            if not isinstance(item, VarRef):
                return
            reference = resolved.resolution.get(item.node_id)
            if reference is not None and reference.decl_node_id in functions:
                referenced.add(reference.decl_node_id)

        # Walking the whole body deliberately includes direct calls, function
        # values, partial applications, and type applications.  Defaults are
        # not body evidence and are checked only by authoritative validation.
        walk(node.body, visit)

        def dependency_key(declaration_id: int) -> tuple[int, int]:
            return _function_key(functions, declaration_id)

        dependencies[declaration_id] = tuple(sorted(referenced, key=dependency_key))
    return dependencies


def infer_module_component_candidates(component: ModuleCandidateComponent) -> None:
    """Infer and close same-module monomorphic function dependency components.

    ``sccs`` returns sink components first for this dependency graph, so every
    acyclic callee becomes a concrete signature before a dependent body can
    consume it. Only one genuinely recursive component shares live flexible
    result variables.
    """
    for module in component.modules:
        functions = _candidate_functions(module)
        dependencies = _function_dependencies(functions, module.resolved)

        def component_key(declaration_id: int) -> tuple[int, int]:
            return _function_key(functions, declaration_id)

        for declaration_ids in sccs(dependencies, key=component_key):
            functions_in_component = tuple(functions[node_id] for node_id in declaration_ids)
            _infer_function_component(module, functions_in_component)


def _infer_function_component(module: CandidateModule, functions: tuple[FuncDef, ...]) -> None:
    """Infer one dependency component and publish only its closed signatures."""
    from agm.agl.typecheck.checker import _Checker

    engine = InferenceEngine()
    session = CandidateSession(engine, frozenset(node.node_id for node in functions))
    provisional: list[tuple[FuncDef, InferenceVarType, FunctionSignature]] = []

    # Make every component member visible before any body is traversed. A
    # dependency component has already closed before this point, whereas peers
    # intentionally share this component's engine.
    for node in functions:
        checker = _Checker(
            env=module.env,
            resolved=module.resolved,
            capabilities=module.capabilities,
            module_id=module.module_id,
        )
        checker._validate_funcdef_header(node)
        result = engine.fresh(f"{node.name} result")
        signature, function_type = resolve_function_header(module.env, node, result_type=result)
        module.env.register_function_signature(node.name, signature)
        module.env.register_function_signature_by_node_id(node.node_id, signature)
        module.env.set_binding_type(node.node_id, function_type)
        provisional.append((node, result, signature))

    session.binding_snapshot = module.env.snapshot_binding_types()
    try:
        for node, result, signature in provisional:
            checker = _Checker(
                env=module.env,
                resolved=module.resolved,
                capabilities=module.capabilities,
                module_id=module.module_id,
            )
            candidate_type = checker.check_candidate_funcdef_body(node, signature, session)
            try:
                engine.unify(
                    result,
                    candidate_type,
                    engine.origin(
                        node.span,
                        role=ConstraintRole.EXPECTED_RESULT,
                        subject=f"return type of function '{node.name}'",
                    ),
                )
            except InferenceError as exc:
                raise AglTypeError(
                    f"Cannot infer return type of function '{node.name}': return values have "
                    "incompatible types. Add a return type annotation.",
                    span=node.span,
                    related=exc.related,
                ) from exc
    finally:
        assert session.binding_snapshot is not None
        module.env.restore_binding_types(session.binding_snapshot)

    unresolved = [
        node
        for node, result, _signature in provisional
        if not engine.is_solved(result) or contains_inference_var(engine.zonk(result))
    ]
    if len(unresolved) == 1:
        node = unresolved[0]
        raise AglTypeError(
            f"Cannot infer return type of function '{node.name}': insufficient concrete return "
            "evidence. Add a return type annotation.",
            span=node.span,
        )
    if unresolved:
        names = ", ".join(f"'{node.name}'" for node in unresolved)
        raise AglTypeError(
            f"Cannot infer return types for functions {names}: insufficient concrete return "
            "evidence. Add return type annotations.",
            span=unresolved[0].span,
        )

    for node, result, signature in provisional:
        concrete_result = engine.zonk(result)
        concrete_signature = FunctionSignature(
            params=signature.params,
            result=concrete_result,
            type_params=signature.type_params,
        )
        module.env.register_function_signature(node.name, concrete_signature)
        module.env.register_function_signature_by_node_id(node.node_id, concrete_signature)
        module.env.set_binding_type(
            node.node_id,
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
