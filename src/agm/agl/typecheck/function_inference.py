"""Function-header resolution and disposable candidate return inference.

Standalone and program checking share this seam. Candidate discovery builds
same-module dependency SCCs for unannotated functions, closes each component
before its dependents, and publishes only concrete signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.persistent import PersistentDict
from agm.agl.semantics.types import (
    FunctionType,
    InferenceVarType,
    Type,
    TypeVarType,
    contains_inference_var,
    substitute,
)
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


@dataclass(frozen=True, slots=True)
class GenericCandidateEdge:
    """One provisional generic occurrence inside a candidate function component."""

    caller_declaration_id: int
    callee_declaration_id: int
    type_args: tuple[Type, ...]
    result: Type
    span: SourceSpan


@dataclass(slots=True)
class CandidateSession:
    """Disposable state shared while discovering one candidate component.

    Candidate checkers use this session's engine and retain their expression
    side tables only for body traversal. The final checker creates all
    published artifacts after every component result is concrete.
    """

    engine: InferenceEngine
    provisional_declaration_ids: frozenset[int]
    evidence: list[SourceSpan] = field(default_factory=list)
    binding_snapshot: PersistentDict[int, Type] | None = None
    current_declaration_id: int | None = None
    generic_edges: list[GenericCandidateEdge] = field(default_factory=list)

    def record_generic_edge(
        self,
        *,
        callee_declaration_id: int,
        type_args: tuple[Type, ...],
        result: Type,
        span: SourceSpan,
    ) -> None:
        """Retain a generic occurrence for component-wide uniformity validation."""
        caller_declaration_id = self.current_declaration_id
        assert caller_declaration_id is not None
        self.generic_edges.append(
            GenericCandidateEdge(
                caller_declaration_id,
                callee_declaration_id,
                type_args,
                result,
                span,
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateModule:
    """One module participating in disposable candidate discovery."""

    resolved: "ModuleResolution"
    env: TypeEnvironment
    capabilities: "HostCapabilities"
    module_id: ModuleId


@dataclass(frozen=True, slots=True)
class ModuleCandidateComponent:
    """Modules whose candidate discovery is coordinated together.

    Current standalone and program paths create synthetic singleton components;
    candidate dependency SCCs are therefore confined to one module.
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
    """Internal declared-signature metadata keyed by a declaration node id.

    Program pre-passes currently create records only for explicit results.
    ``candidate_evidence`` remains reserved for candidate metadata, whose
    inferred signatures are currently published only in their own module.
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
    """Return unannotated ordinary functions by declaration id."""
    program = module.resolved.program
    assert isinstance(program, Program)
    return {
        item.node_id: item
        for item in program.body.items
        if isinstance(item, FuncDef)
        and item.return_type is None
        and not item.is_builtin
        and not item.is_extern
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
    """Infer and close same-module function dependency components.

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

    _close_generic_candidate_edges(session, provisional)

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


def _close_generic_candidate_edges(
    session: CandidateSession,
    provisional: list[tuple[FuncDef, InferenceVarType, FunctionSignature]],
) -> None:
    """Validate uniform generic recursion and connect its delayed result evidence."""
    functions = {node.node_id: (node, result) for node, result, _signature in provisional}
    engine = session.engine
    for edge in session.generic_edges:
        caller, _ = functions[edge.caller_declaration_id]
        callee, _ = functions[edge.callee_declaration_id]
        caller_vector = tuple(TypeVarType(name) for name in caller.type_params)
        type_args = tuple(engine.zonk(arg) for arg in edge.type_args)
        if len(type_args) != len(caller_vector) or type_args != caller_vector:
            raise AglTypeError(
                f"Cannot infer return type of function '{caller.name}': recursive call to "
                f"'{callee.name}' changes its generic type arguments. Add a return type "
                "annotation.",
                span=edge.span,
            )

    for edge in session.generic_edges:
        callee, callee_result = functions[edge.callee_declaration_id]
        result_template = engine.zonk(callee_result)
        substitutions = dict(
            zip(callee.type_params, (engine.zonk(arg) for arg in edge.type_args), strict=True)
        )
        result = substitute(result_template, substitutions)
        try:
            engine.unify(
                edge.result,
                result,
                engine.origin(
                    edge.span,
                    role=ConstraintRole.EXPECTED_RESULT,
                    subject=f"recursive call to '{callee.name}'",
                ),
            )
        except InferenceError as exc:
            raise AglTypeError(
                f"Cannot infer return type of function '{callee.name}': recursive return "
                "evidence is incompatible. Add a return type annotation.",
                span=edge.span,
                related=exc.related,
            ) from exc


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
