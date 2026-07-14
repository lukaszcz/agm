"""Checker-independent first-order inference for AgL semantic types.

This module owns solver-local flexible variables, fresh scheme instantiation,
exact equality constraints, contextual completion, final zonking, solve
requirements, and constraint provenance.  It deliberately has no assignability,
coercion, lowering, runtime, or checker-side-table dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from agm.agl.diagnostics import AglError
from agm.agl.semantics.types import (
    BottomType,
    DictType,
    EnumType,
    FunctionType,
    InferenceVarType,
    ListType,
    RecordType,
    Type,
    TypeVarType,
    type_children,
)
from agm.agl.syntax.spans import SourceSpan


class ConstraintRole(StrEnum):
    """The semantic source of an equality or contextual constraint."""

    FUNCTION_ARGUMENT = "function argument"
    CONSTRUCTOR_FIELD = "constructor field"
    EXPECTED_RESULT = "expected result"
    PARTIAL_HOLE = "partial hole"
    LITERAL_ELEMENT = "literal element"
    EXPLICIT_INSTANTIATION = "explicit instantiation"


@dataclass(frozen=True, slots=True)
class ConstraintOrigin:
    """Stable, user-facing provenance for one solver constraint."""

    span: SourceSpan
    sequence: int
    role: ConstraintRole
    subject: str
    type_param: str | None = None


@dataclass(frozen=True, slots=True)
class SchemeInstantiation:
    """Fresh monotype templates and their ordered quantified-variable map."""

    templates: tuple[Type, ...]
    variables: dict[str, InferenceVarType]


class InferenceError(AglError):
    """A source-aware failure of exact equality inference."""


@dataclass(frozen=True, slots=True)
class _SolveRequirement:
    variable: InferenceVarType
    origin: ConstraintOrigin


class InferenceEngine:
    """A provenance-aware, first-order unifier for one inference region.

    Flexible variables form a deterministic union-find forest.  A root may
    additionally have one structural solution.  Rigid source variables are
    ordinary leaves: they can only equal themselves, while a flexible variable
    may solve to one.  The engine intentionally knows nothing about AgL's
    assignability rules; callers apply those after :meth:`zonk`.
    """

    def __init__(self) -> None:
        self._parent: dict[InferenceVarType, InferenceVarType] = {}
        self._solution: dict[InferenceVarType, Type] = {}
        self._evidence: dict[InferenceVarType, tuple[ConstraintOrigin, ...]] = {}
        self._order: dict[InferenceVarType, int] = {}
        self._next_variable = 0
        self._next_origin = 0
        self._requirements: list[_SolveRequirement] = []

    def origin(
        self,
        span: SourceSpan,
        *,
        role: ConstraintRole,
        subject: str,
        type_param: str | None = None,
    ) -> ConstraintOrigin:
        """Create the next stable origin in this inference region."""
        origin = ConstraintOrigin(span, self._next_origin, role, subject, type_param)
        self._next_origin += 1
        return origin

    def fresh(self, display_hint: str = "") -> InferenceVarType:
        """Allocate one flexible variable owned by this engine."""
        variable = InferenceVarType(display_hint)
        self._register_variable(variable)
        return variable

    def instantiate(
        self, type_params: Sequence[str], templates: Sequence[Type]
    ) -> SchemeInstantiation:
        """Freshen ordered quantified templates without changing other rigids."""
        variables = {name: self.fresh(name) for name in type_params}
        return SchemeInstantiation(
            templates=tuple(self._freshen(template, variables) for template in templates),
            variables=variables,
        )

    def unify(self, left: Type, right: Type, origin: ConstraintOrigin) -> None:
        """Require exact structural equality, retaining ``origin`` as evidence."""
        self._unify(left, right, origin, ())

    def complete_from_context(
        self, inferred: Type, context: Type, origin: ConstraintOrigin
    ) -> None:
        """Use matching context to fill only unresolved inferred representatives.

        Unlike :meth:`unify`, this never reports shape mismatches and never
        changes a flexible representative already solved by equality evidence.
        """
        self._complete(inferred, context, origin)

    def resolve(self, typ: Type) -> Type:
        """Return ``typ`` with links followed and compressed."""
        return self.zonk(typ)

    def zonk(self, typ: Type) -> Type:
        """Recursively replace flexible links with their final known solutions."""
        if isinstance(typ, InferenceVarType):
            root = self._find(typ)
            solution = self._solution.get(root)
            if solution is None:
                return root
            zonked = self.zonk(solution)
            self._solution[root] = zonked
            return zonked
        if isinstance(typ, ListType):
            return ListType(self.zonk(typ.elem))
        if isinstance(typ, DictType):
            return DictType(self.zonk(typ.value))
        if isinstance(typ, FunctionType):
            return FunctionType(
                tuple(self.zonk(param) for param in typ.params), self.zonk(typ.result)
            )
        if isinstance(typ, RecordType):
            return RecordType(
                typ.name, tuple(self.zonk(arg) for arg in typ.type_args), typ.module_id
            )
        if isinstance(typ, EnumType):
            return EnumType(typ.name, tuple(self.zonk(arg) for arg in typ.type_args), typ.module_id)
        return typ

    def is_solved(self, variable: InferenceVarType) -> bool:
        """Return whether ``variable`` has a fully resolved final solution."""
        return not self._contains_unresolved_variable(variable)

    def parent_of(self, variable: InferenceVarType) -> InferenceVarType:
        """Return the compressed union-find representative for testable introspection."""
        return self._find(variable)

    def require_solved(self, variable: InferenceVarType, origin: ConstraintOrigin) -> None:
        """Register an ordered requirement that this quantified variable resolves."""
        self._find(variable)
        self._requirements.append(_SolveRequirement(variable, origin))

    def check_requirements(self) -> None:
        """Reject the first registered quantified variable that remains unresolved."""
        for requirement in self._requirements:
            if not self.is_solved(requirement.variable):
                origin = requirement.origin
                if origin.type_param is None:
                    raise InferenceError(
                        f"Cannot infer type of {origin.subject}; add an explicit container "
                        "type annotation.",
                        span=origin.span,
                    )
                raise InferenceError(
                    f"Cannot infer type arguments for '{origin.subject}': "
                    f"type argument '{origin.type_param}' remains unresolved; "
                    f"supply it explicitly via '{origin.subject}::[…]'.",
                    span=origin.span,
                )

    def assert_no_inference_vars(self, *types: Type) -> None:
        """Assert that finalized types contain no flexible inference variables."""
        for typ in types:
            if self._contains_unresolved_variable(typ):
                raise AssertionError("inference variable leaked from a finalized inference region")

    def assert_no_owned_leaks(self, types: Iterable[Type]) -> None:
        """Assert the reusable finalization invariant over an iterable of types."""
        self.assert_no_inference_vars(*tuple(types))

    def _freshen(self, typ: Type, variables: dict[str, InferenceVarType]) -> Type:
        if isinstance(typ, TypeVarType):
            return variables.get(typ.name, typ)
        if isinstance(typ, ListType):
            return ListType(self._freshen(typ.elem, variables))
        if isinstance(typ, DictType):
            return DictType(self._freshen(typ.value, variables))
        if isinstance(typ, FunctionType):
            return FunctionType(
                tuple(self._freshen(param, variables) for param in typ.params),
                self._freshen(typ.result, variables),
            )
        if isinstance(typ, RecordType):
            return RecordType(
                typ.name,
                tuple(self._freshen(arg, variables) for arg in typ.type_args),
                typ.module_id,
            )
        if isinstance(typ, EnumType):
            return EnumType(
                typ.name,
                tuple(self._freshen(arg, variables) for arg in typ.type_args),
                typ.module_id,
            )
        return typ

    def _unify(
        self,
        left: Type,
        right: Type,
        origin: ConstraintOrigin,
        inherited: tuple[ConstraintOrigin, ...],
    ) -> None:
        evidence = self._merge_origins(inherited, self._origins_in(left), self._origins_in(right))
        left = self.zonk(left)
        right = self.zonk(right)
        if left == right:
            self._add_evidence_from_type(left, origin)
            return
        if isinstance(left, BottomType) or isinstance(right, BottomType):
            return
        if isinstance(left, InferenceVarType):
            self._bind(left, right, origin)
            return
        if isinstance(right, InferenceVarType):
            self._bind(right, left, origin)
            return
        if isinstance(left, ListType) and isinstance(right, ListType):
            self._unify(left.elem, right.elem, origin, evidence)
            return
        if isinstance(left, DictType) and isinstance(right, DictType):
            self._unify(left.value, right.value, origin, evidence)
            return
        if isinstance(left, FunctionType) and isinstance(right, FunctionType):
            if len(left.params) != len(right.params):
                self._raise_mismatch(left, right, origin, evidence)
            for left_param, right_param in zip(left.params, right.params, strict=True):
                self._unify(left_param, right_param, origin, evidence)
            self._unify(left.result, right.result, origin, evidence)
            return
        if isinstance(left, RecordType) and isinstance(right, RecordType):
            if left.name != right.name or left.module_id != right.module_id:
                self._raise_mismatch(left, right, origin, evidence)
            self._unify_nominal_args(left.type_args, right.type_args, left, right, origin, evidence)
            return
        if isinstance(left, EnumType) and isinstance(right, EnumType):
            if left.name != right.name or left.module_id != right.module_id:
                self._raise_mismatch(left, right, origin, evidence)
            self._unify_nominal_args(left.type_args, right.type_args, left, right, origin, evidence)
            return
        self._raise_mismatch(left, right, origin, evidence)

    def _unify_nominal_args(
        self,
        left_args: tuple[Type, ...],
        right_args: tuple[Type, ...],
        left: Type,
        right: Type,
        origin: ConstraintOrigin,
        evidence: tuple[ConstraintOrigin, ...],
    ) -> None:
        if len(left_args) != len(right_args):
            self._raise_mismatch(left, right, origin, evidence)
        for left_arg, right_arg in zip(left_args, right_args, strict=True):
            self._unify(left_arg, right_arg, origin, evidence)

    def _bind(self, variable: InferenceVarType, typ: Type, origin: ConstraintOrigin) -> None:
        variable = self._find(variable)
        typ = self.zonk(typ)
        if isinstance(typ, InferenceVarType):
            self._merge(variable, self._find(typ), origin)
            return
        if self._occurs(variable, typ):
            evidence = self._merge_origins(self._evidence[variable], self._origins_in(typ))
            self._raise_infinite(variable, typ, origin, evidence)
        self._solution[variable] = typ
        self._add_evidence(variable, origin)

    def _merge(
        self,
        first: InferenceVarType,
        second: InferenceVarType,
        origin: ConstraintOrigin,
    ) -> None:
        if self._order[first] > self._order[second]:
            first, second = second, first
        self._parent[second] = first
        self._evidence[first] = self._merge_origins(
            self._evidence[first], self._evidence[second], (origin,)
        )
        self._evidence.pop(second)

    def _complete(self, inferred: Type, context: Type, origin: ConstraintOrigin) -> None:
        inferred = self.zonk(inferred)
        context = self.zonk(context)
        if isinstance(inferred, BottomType) or isinstance(context, BottomType):
            return
        if isinstance(inferred, InferenceVarType):
            if isinstance(context, InferenceVarType) and inferred == context:
                return
            if not self._occurs(inferred, context):
                self._bind(inferred, context, origin)
            return
        if isinstance(inferred, ListType) and isinstance(context, ListType):
            self._complete(inferred.elem, context.elem, origin)
            return
        if isinstance(inferred, DictType) and isinstance(context, DictType):
            self._complete(inferred.value, context.value, origin)
            return
        if isinstance(inferred, FunctionType) and isinstance(context, FunctionType):
            if len(inferred.params) != len(context.params):
                return
            for inferred_param, context_param in zip(inferred.params, context.params, strict=True):
                self._complete(inferred_param, context_param, origin)
            self._complete(inferred.result, context.result, origin)
            return
        if isinstance(inferred, RecordType) and isinstance(context, RecordType):
            if inferred.name == context.name and inferred.module_id == context.module_id:
                self._complete_nominal_args(inferred.type_args, context.type_args, origin)
            return
        if isinstance(inferred, EnumType) and isinstance(context, EnumType):
            if inferred.name == context.name and inferred.module_id == context.module_id:
                self._complete_nominal_args(inferred.type_args, context.type_args, origin)
            return

    def _complete_nominal_args(
        self,
        inferred_args: tuple[Type, ...],
        context_args: tuple[Type, ...],
        origin: ConstraintOrigin,
    ) -> None:
        if len(inferred_args) != len(context_args):
            return
        for inferred_arg, context_arg in zip(inferred_args, context_args, strict=True):
            self._complete(inferred_arg, context_arg, origin)

    def _register_variable(self, variable: InferenceVarType) -> None:
        self._parent[variable] = variable
        self._evidence[variable] = ()
        self._order[variable] = self._next_variable
        self._next_variable += 1

    def _find(self, variable: InferenceVarType) -> InferenceVarType:
        if variable not in self._parent:
            raise AssertionError("inference variable is not owned by this engine")
        parent = self._parent[variable]
        if parent != variable:
            parent = self._find(parent)
            self._parent[variable] = parent
        return parent

    def _occurs(self, variable: InferenceVarType, typ: Type) -> bool:
        variable = self._find(variable)
        if isinstance(typ, InferenceVarType):
            return self._find(typ) == variable
        return any(self._occurs(variable, child) for child in type_children(typ))

    def _origins_in(self, typ: Type) -> tuple[ConstraintOrigin, ...]:
        if isinstance(typ, InferenceVarType):
            return self._evidence[self._find(typ)]
        return self._merge_origins(*(self._origins_in(child) for child in type_children(typ)))

    def _add_evidence_from_type(self, typ: Type, origin: ConstraintOrigin) -> None:
        if isinstance(typ, InferenceVarType):
            self._add_evidence(typ, origin)
        for child in type_children(typ):
            self._add_evidence_from_type(child, origin)

    def _add_evidence(self, variable: InferenceVarType, origin: ConstraintOrigin) -> None:
        root = self._find(variable)
        self._evidence[root] = self._merge_origins(self._evidence[root], (origin,))

    def _merge_origins(self, *groups: tuple[ConstraintOrigin, ...]) -> tuple[ConstraintOrigin, ...]:
        origins = {origin for group in groups for origin in group}
        return tuple(sorted(origins, key=self._origin_key))

    def _origin_key(
        self, origin: ConstraintOrigin
    ) -> tuple[int, str, int, int, int, int, str, str, str]:
        span = origin.span
        return (
            origin.sequence,
            span.source.label,
            span.start_offset,
            span.end_offset,
            span.start_line,
            span.start_col,
            origin.role.value,
            origin.subject,
            origin.type_param or "",
        )

    def _raise_mismatch(
        self,
        left: Type,
        right: Type,
        origin: ConstraintOrigin,
        evidence: tuple[ConstraintOrigin, ...],
    ) -> None:
        self._raise_error(f"Cannot unify {left!r} with {right!r}.", origin, evidence)

    def _raise_infinite(
        self,
        variable: InferenceVarType,
        typ: Type,
        origin: ConstraintOrigin,
        evidence: tuple[ConstraintOrigin, ...],
    ) -> None:
        del variable, typ
        self._raise_error("Cannot infer an infinite type.", origin, evidence)

    def _raise_error(
        self,
        message: str,
        origin: ConstraintOrigin,
        evidence: tuple[ConstraintOrigin, ...],
    ) -> None:
        earlier = tuple(item for item in evidence if item != origin)
        related: tuple[tuple[str, SourceSpan], ...] = ()
        if earlier:
            first = earlier[0]
            label = first.type_param or first.subject
            related_message = (
                f"{label} was first constrained by {first.role.value} '{first.subject}'."
            )
            related = ((related_message, first.span),)
        raise InferenceError(message, span=origin.span, related=related)

    def _contains_unresolved_variable(self, typ: Type) -> bool:
        resolved = self.zonk(typ)
        if isinstance(resolved, InferenceVarType):
            return True
        return any(self._contains_unresolved_variable(child) for child in type_children(resolved))


InferenceSolver = InferenceEngine
"""Backward-compatible descriptive alias for :class:`InferenceEngine`."""
