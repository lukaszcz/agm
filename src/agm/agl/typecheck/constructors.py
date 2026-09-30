"""Constructor (record/enum-variant/exception, generic + cross-module) call/value checker.

Driven by ``_Checker`` via the narrow ``ConstructorCheckCtx`` Protocol.  All
logic lives here; the host checker instantiates ``ConstructorChecker(self)``
and delegates the constructor dispatch branches in ``_check_varref`` and
``_check_call`` to the public entry points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal, Protocol, TypeGuard, cast

from agm.agl.diagnostics import type_name_not_a_value
from agm.agl.scope.symbols import ConstructorRef, ModuleResolution
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    EnumType,
    ExceptionType,
    FunctionType,
    RecordType,
    Type,
    TypeTemplate,
    TypeVarType,
    free_type_vars,
    substitute,
)
from agm.agl.syntax.nodes import (
    Call,
    CallArg,
    Expr,
    NamedArg,
    Placeholder,
    QualifierChain,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr, render_qualified_name
from agm.agl.typecheck.arguments import bind_constructor_args
from agm.agl.typecheck.env import (
    AglTypeError,
    ConstructorSignature,
    TypeEnvironment,
)
from agm.agl.typecheck.inference import ConstraintRole, InferenceEngine
from agm.agl.zones import ParamZone


def _nominal_constructor_signature(
    table: TypeTable,
    owner: RecordType | ExceptionType,
    owner_name: str,
    type_params: tuple[str, ...],
) -> ConstructorSignature:
    """Return the signature of the constructor producing *owner*.

    *owner* applies its declaration to arguments over *type_params*; the
    fields -- an exception's including its bases' -- are read by *owner*'s
    declaration identity with those arguments substituted. *owner_name* is
    the spelling the constructor was written with -- the declaration's own
    name or a transparent alias of it.
    """
    fields = (
        table.record_fields(owner)
        if isinstance(owner, RecordType)
        else table.exception_fields(owner)
    )
    return ConstructorSignature(
        owner_name=owner_name,
        field_names=tuple(fields),
        field_templates=tuple(fields.values()),
        result_template=owner,
        type_params=type_params,
    )


def _constructible_nominal(
    template: Type, spelling: str, span: SourceSpan
) -> RecordType | ExceptionType:
    """Return *template*, the record or exception a constructor written *spelling* denotes.

    Scope presumes an alias constructible when it selects no declaration for
    the alias's target; the checked target decides.
    """
    if not isinstance(template, (RecordType, ExceptionType)):
        raise type_name_not_a_value(spelling, span)
    return template


def constructed_template(env: TypeEnvironment, ctor_ref: ConstructorRef) -> TypeTemplate:
    """Return the type *ctor_ref* constructs, quantified over its type parameters.

    A declaration is read by its identity. An alias is its checked template,
    or the member ``ctor_ref.member`` of that template for an alias of an
    enum; only the alias parameters the result mentions quantify it, since
    a parameter it does not mention is neither inferable nor significant.
    """
    typedef = env.type_table.get_by_id(ctor_ref.owner_decl_node_id)
    if typedef is not None:
        return TypeTemplate(
            typedef.handle(tuple(TypeVarType(param) for param in typedef.type_params)),
            typedef.type_params,
        )
    # Scope publishes an alias constructor only for a declared, checked alias.
    source = env.declared_type_template(
        ctor_ref.owner_module_id, ctor_ref.owner_name, scope_path=ctor_ref.owner_path
    )
    template = source.template
    if ctor_ref.member is not None:
        # Scope selects a member only through an alias whose target is an enum
        # declaring it.
        template = env.type_table.enum_member_names(cast(EnumType, template))[ctor_ref.member]
    mentioned = free_type_vars(template)
    type_params = tuple(param for param in source.type_params if param in mentioned)
    return TypeTemplate(template, type_params)


def constructor_signature_of(
    env: TypeEnvironment, ctor_ref: ConstructorRef, span: SourceSpan, *, spelling: str
) -> tuple[ConstructorRef, ConstructorSignature]:
    """Return *ctor_ref*, written *spelling*, with its type's parameters and its signature.

    Scope owns the source spelling and identity; :func:`constructed_template`
    reads the constructed type from that identity.
    """
    template = constructed_template(env, ctor_ref)
    owner = _constructible_nominal(template.template, spelling, span)
    return replace(ctor_ref, type_params=template.type_params), _nominal_constructor_signature(
        env.type_table, owner, ctor_ref.owner_name, template.type_params
    )


def applies_owner(qualifier: QualifierChain | None) -> TypeGuard[QualifierChain]:
    """Whether *qualifier* applies its final owner to explicit type arguments."""
    return (
        qualifier is not None
        and bool(qualifier.segments)
        and qualifier.segments[-1].type_args is not None
    )


def applied_owner_member(
    env: TypeEnvironment,
    qualifier: QualifierChain | None,
    member: str,
    *,
    type_vars: frozenset[str],
    span: SourceSpan,
) -> RecordType | None:
    """Return the member ``Owner[A]::member`` selects when *qualifier* applies its owner.

    A record's own constructor spelling (``Box[int]::Box``) selects the record
    at those arguments.
    """
    if not applies_owner(qualifier):
        return None
    owner = env.owner_type_for_qualifier(qualifier, span=span, type_vars=type_vars)
    if owner is not None and isinstance(owner[0], RecordType):
        return owner[0]
    selected = env.select_owner_inline_member(qualifier, member, type_vars=type_vars, span=span)
    return None if selected is None else selected.member


def selected_constructor_signature(
    env: TypeEnvironment,
    ref: VarRef,
    ctor_ref: ConstructorRef,
    span: SourceSpan,
    *,
    type_vars: frozenset[str],
) -> tuple[ConstructorRef, ConstructorSignature]:
    """Return the constructor *ctor_ref* that *ref* spells, and its signature.

    An applied owner ``Owner[A]::Member`` selects the member concretely at
    those arguments; any other spelling constructs what *ctor_ref* identifies.
    """
    member = applied_owner_member(
        env, ref.qualifier, ctor_ref.member or ctor_ref.owner_name, type_vars=type_vars, span=span
    )
    if member is None:
        return constructor_signature_of(
            env, ctor_ref, span, spelling=render_qualified_name(ref.qualifier, ref.name)
        )
    return replace(ctor_ref, type_params=()), _nominal_constructor_signature(
        env.type_table, member, ctor_ref.owner_name, ()
    )


# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class ConstructorCheckCtx(Protocol):
    """The minimal _Checker surface the constructor checker needs."""

    _env: TypeEnvironment
    _resolved: ModuleResolution
    _current_type_vars: frozenset[str]

    def _record_constructor_call_binding(self, node_id: int, binding: dict[str, Expr]) -> None: ...

    def _record_partial_call(
        self,
        node: Call,
        binding: Sequence[CallArg | None],
        hole_indices: Mapping[int, int],
        *,
        callee_kind: Literal["declared", "constructor", "value"] = "declared",
    ) -> None: ...

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type: ...

    def _assert_assignable_from(
        self, value_type: Type, target_type: Type, span: SourceSpan, expr: Expr
    ) -> None: ...

    def _constrain_argument(
        self,
        slot_type: Type,
        arg_expr: Expr,
        *,
        role: ConstraintRole,
        subject: str,
        error_subject: str,
    ) -> Type: ...

    def _instantiate_generic_constructor_value(
        self,
        *,
        type_params: tuple[str, ...],
        field_templates: tuple[Type, ...],
        result_template: Type,
        span: SourceSpan,
        expected: Type | None,
        subject: str,
    ) -> Type: ...

    def _zonk_constructor_owner[N: RecordType | EnumType | ExceptionType](self, owner: N) -> N: ...

    def _active_inference_engine(self) -> InferenceEngine: ...

    def _set_generic_constructor_result_provenance(
        self,
        node_id: int,
        result_template: Type,
        field_templates: Mapping[str, Type],
        bound_exprs: Mapping[str, CallArg],
    ) -> None: ...

    def _frame_generic_constraint_error(
        self, exc: AglTypeError, exprs: tuple[Expr, ...]
    ) -> AglTypeError: ...


# ---------------------------------------------------------------------------
# Collaborator class
# ---------------------------------------------------------------------------


class ConstructorChecker:
    """Type-checking collaborator for constructor call and value nodes.

    Instantiated once per ``_Checker`` instance (``self._constructors``).
    Handles record, enum-variant, exception, generic, and cross-module
    constructor checking; ``_Checker`` delegates the relevant branches in
    ``_check_varref`` and ``_check_call`` here.
    """

    def __init__(self, ctx: ConstructorCheckCtx) -> None:
        self._ctx = ctx

    # --- Generic constructor as value ---

    def check_generic_constructor_as_value(
        self,
        *,
        ctor_ref: ConstructorRef,
        span: SourceSpan,
        expected: Type | None,
        sig: ConstructorSignature,
    ) -> Type:
        """Handle a generic constructor used as a bare value (not in direct call position).

        For nullary variants (no fields): instantiate from the expected nominal type.
        For payload constructors: instantiate to a FunctionType from expected FunctionType.
        """
        result = self._ctx._instantiate_generic_constructor_value(
            type_params=ctor_ref.type_params,
            field_templates=sig.field_templates,
            result_template=sig.result_template,
            span=span,
            expected=expected,
            subject=ctor_ref.owner_name,
        )
        return self._contextualize_member_result(result, expected, span, ctor_ref.owner_name)

    def _contextualize_member_result(
        self, result: Type, expected: Type | None, span: SourceSpan, owner_name: str
    ) -> Type:
        """Apply an enum result context to a constructor function value only."""
        if (
            isinstance(result, FunctionType)
            and isinstance(expected, FunctionType)
            and self._unify_member_with_expected(result.result, expected.result, span, owner_name)
        ):
            return replace(result, result=expected.result)
        return result

    def _unify_member_with_expected(
        self, result: Type, expected: Type | None, span: SourceSpan, owner_name: str
    ) -> bool:
        """Specialize a member constructor's record result against an expected enum slot.

        Returns whether *result* is a member of *expected* and was unified with
        the enum's own instantiation of it -- the step that lets an expected
        enum type settle the member's type arguments.
        """
        if not isinstance(result, RecordType) or not isinstance(expected, EnumType):
            return False
        expected_member = self._ctx._env.type_table.enum_member_by_decl(expected, result.decl_id)
        if expected_member is None:
            return False
        engine = self._inference_engine()
        engine.unify(
            result,
            expected_member,
            engine.origin(span, role=ConstraintRole.EXPECTED_RESULT, subject=owner_name),
        )
        return True

    # --- Generic constructor type-apply as value (explicit type args) ---

    def _instantiate_constructor_value(
        self,
        *,
        type_params: tuple[str, ...],
        type_args: tuple[TypeExpr, ...],
        sig: ConstructorSignature,
        span: SourceSpan,
    ) -> Type:
        """Instantiate a generic constructor value from explicit type arguments.

        Shared core of the bare and qualified type-apply-as-value paths. A
        field-bearing constructor yields a ``FunctionType`` from its field
        types to its concrete member record; a fieldless one constructs that
        record immediately.
        """
        subst = {
            p: self._ctx._env.resolve_type_expr(
                ta, span=span, type_vars=self._ctx._current_type_vars
            )
            for p, ta in zip(type_params, type_args)
        }
        concrete_params = tuple(substitute(ft, subst) for ft in sig.field_templates)
        concrete_result = substitute(sig.result_template, subst)
        if not concrete_params:
            return concrete_result
        return FunctionType(params=concrete_params, result=concrete_result)

    def _inline_enum_owner_type_params(self, ctor_ref: ConstructorRef) -> tuple[str, ...] | None:
        """Return the generic owner parameters for an inline member constructor.

        A direct member reference normally applies just the parameters captured
        by that member record. Its enclosing enum is a second, unambiguous
        application target only when the member's captured parameters agree
        with the selected constructor metadata.
        """
        if not ctor_ref.owner_path:
            return None
        enum_name = ctor_ref.owner_path[-1]
        enum_path = ctor_ref.owner_path[:-1]
        enum_gdef = self._ctx._env.get_generic_type_from_module(
            ctor_ref.owner_module_id, enum_name, scope_path=enum_path
        )
        if enum_gdef is None:
            scoped_name = "::".join((*enum_path, enum_name))
            enum_gdef = self._ctx._env.get_generic_type(scoped_name)
        if enum_gdef is None or not isinstance(enum_gdef.template, EnumType):
            return None
        member = self._ctx._env.type_table.enum_member_by_decl(
            enum_gdef.template, ctor_ref.owner_decl_node_id
        )
        if member is None:
            return None
        captured = tuple(
            argument.name for argument in member.type_args if isinstance(argument, TypeVarType)
        )
        return enum_gdef.type_params if captured == ctor_ref.type_params else None

    def _explicit_constructor_type_params(
        self, ctor_ref: ConstructorRef, type_args: tuple[TypeExpr, ...], span: SourceSpan
    ) -> tuple[str, ...]:
        """Select direct-member or owner parameters for explicit type arguments.

        An alias constructor takes the alias's declared parameters.
        """
        if len(type_args) == len(ctor_ref.type_params):
            return ctor_ref.type_params
        owner_type_params = self._inline_enum_owner_type_params(ctor_ref)
        if owner_type_params is not None and len(type_args) == len(owner_type_params):
            return owner_type_params
        if not ctor_ref.type_params and owner_type_params is None:
            raise AglTypeError(
                f"'{ctor_ref.owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        raise AglTypeError(
            f"'{ctor_ref.owner_name}' requires {len(ctor_ref.type_params)} type argument(s), "
            f"but {len(type_args)} were supplied.",
            span=span,
        )

    def check_constructor_type_apply(
        self,
        *,
        ref: VarRef,
        ctor_ref: ConstructorRef,
        type_args: tuple[TypeExpr, ...],
        span: SourceSpan,
        expected: Type | None,
    ) -> Type:
        """Type the constructor *ref* spells, explicitly instantiated and used as a value.

        Direct members accept their captured parameters.  An inline generic
        enum member also accepts its owner's complete parameter list, which is
        substituted through the member's captured result and field templates.
        """
        type_params = self._explicit_constructor_type_params(ctor_ref, type_args, span)
        ctor_ref, sig = constructor_signature_of(
            self._ctx._env, ctor_ref, span, spelling=render_qualified_name(ref.qualifier, ref.name)
        )
        result = self._instantiate_constructor_value(
            type_params=type_params,
            type_args=type_args,
            sig=sig,
            span=span,
        )
        return self._contextualize_member_result(result, expected, span, ctor_ref.owner_name)

    # --- Generic constructor call (private helper) ---

    def _check_generic_constructor_call(
        self,
        *,
        node_type_args: tuple[TypeExpr, ...],
        ctor_ref: ConstructorRef,
        node: Call,
        expected: Type | None,
        hole_indices: Mapping[int, int],
        sig: ConstructorSignature,
    ) -> Type:
        """Check a generic constructor call through the expression-region solver."""
        span = node.span
        owner_name = ctor_ref.owner_name
        type_params = ctor_ref.type_params
        field_kinds = self._ctx._env.type_table.field_kinds(sig.result_template)
        bound_exprs = bind_constructor_args(
            field_kinds,
            node.args,
            node.named_args,
            call_span=span,
            context_desc=f"constructor '{owner_name}'",
        )

        if node_type_args:
            subst = {
                type_param: self._ctx._env.resolve_type_expr(
                    type_arg, span=span, type_vars=self._ctx._current_type_vars
                )
                for type_param, type_arg in zip(type_params, node_type_args, strict=True)
            }
            field_types = tuple(substitute(template, subst) for template in sig.field_templates)
            result = substitute(sig.result_template, subst)
        else:
            engine = self._inference_engine()
            instantiation = engine.instantiate(
                type_params, (*sig.field_templates, sig.result_template)
            )
            field_types = instantiation.templates[:-1]
            result = instantiation.templates[-1]
            self._unify_member_with_expected(result, expected, span, owner_name)
            for type_param in type_params:
                engine.require_solved(
                    instantiation.variables[type_param],
                    engine.origin(
                        span,
                        role=ConstraintRole.EXPECTED_RESULT,
                        subject=owner_name,
                        type_param=type_param,
                    ),
                )

        fields_by_name = dict(zip(sig.field_names, field_types, strict=True))
        participants = tuple(
            expr for expr in bound_exprs.values() if not isinstance(expr, Placeholder)
        )
        try:
            for field_name, _field_kind in field_kinds:
                bound_expr = bound_exprs[field_name]
                if isinstance(bound_expr, Placeholder):
                    continue
                self._ctx._constrain_argument(
                    fields_by_name[field_name],
                    bound_expr,
                    role=ConstraintRole.CONSTRUCTOR_FIELD,
                    subject=field_name,
                    error_subject=f"constructor '{owner_name}'",
                )
        except AglTypeError as exc:
            framed = self._ctx._frame_generic_constraint_error(exc, participants)
            if framed is exc:
                raise
            raise framed from exc

        produced = self._constructor_call_result_type(
            field_kinds, fields_by_name, result, bound_exprs, hole_indices
        )
        contextualized = self._contextualize_member_result(produced, expected, span, owner_name)
        if expected is not None and not node_type_args:
            engine = self._inference_engine()
            if contextualized is produced:
                engine.complete_from_context(
                    produced,
                    expected,
                    engine.origin(span, role=ConstraintRole.EXPECTED_RESULT, subject=owner_name),
                )
        self._record_call_binding(node, bound_exprs, hole_indices)
        if not node_type_args:
            self._ctx._set_generic_constructor_result_provenance(
                node.node_id,
                sig.result_template,
                dict(zip(sig.field_names, sig.field_templates, strict=True)),
                bound_exprs,
            )
        return contextualized

    def _inference_engine(self) -> InferenceEngine:
        """Return the active shared solver for a generic constructor occurrence."""
        return self._ctx._active_inference_engine()

    # --- Constructor call helpers ---

    def _record_call_binding(
        self, node: Call, bound_exprs: Mapping[str, CallArg], hole_indices: Mapping[int, int]
    ) -> None:
        """Record a complete call's field binding, or a partial call's."""
        supplied = {
            name: arg for name, arg in bound_exprs.items() if not isinstance(arg, Placeholder)
        }
        if len(supplied) == len(bound_exprs):
            self._ctx._record_constructor_call_binding(node.node_id, supplied)
        else:
            self._ctx._record_partial_call(
                node, tuple(bound_exprs.values()), hole_indices, callee_kind="constructor"
            )

    def _constructor_fields_and_context(
        self, owner: RecordType | ExceptionType
    ) -> tuple[Mapping[str, Type], str]:
        owner = self._ctx._zonk_constructor_owner(owner)
        if isinstance(owner, RecordType):
            return self._ctx._env.type_table.record_fields(owner), f"constructor '{owner.name}'"
        return self._ctx._env.type_table.exception_fields(owner), f"exception '{owner.name}'"

    @staticmethod
    def _constructor_call_result_type(
        field_kinds: tuple[tuple[str, ParamZone], ...],
        field_types: Mapping[str, Type],
        result: Type,
        bound_exprs: Mapping[str, CallArg],
        hole_indices: Mapping[int, int],
    ) -> Type:
        if not hole_indices:
            return result
        hole_types = {
            hole_indices[bound_expr.node_id]: field_types[fname]
            for fname, _fkind in field_kinds
            if isinstance(bound_expr := bound_exprs[fname], Placeholder)
        }
        return FunctionType(
            params=tuple(hole_types[index] for index in range(len(hole_indices))),
            result=result,
        )

    def _finish_constructor_call(
        self,
        *,
        owner: RecordType | ExceptionType,
        field_kinds: tuple[tuple[str, ParamZone], ...],
        bound_exprs: Mapping[str, CallArg],
        node: Call | None,
        hole_indices: Mapping[int, int],
    ) -> Type:
        owner = self._ctx._zonk_constructor_owner(owner)
        fields, _context_desc = self._constructor_fields_and_context(owner)

        if node is not None:
            self._record_call_binding(node, bound_exprs, hole_indices)

        # Type-check each supplied field. Placeholder fields are checked when
        # the produced function is invoked.
        for fname, _fkind in field_kinds:
            expected_field_type = fields[fname]
            arg_expr = bound_exprs[fname]
            if isinstance(arg_expr, Placeholder):
                continue
            arg_type = self._ctx._check_expr(arg_expr, expected=expected_field_type)
            self._ctx._assert_assignable_from(
                arg_type, expected_field_type, arg_expr.span, arg_expr
            )

        return self._constructor_call_result_type(
            field_kinds, fields, owner, bound_exprs, hole_indices
        )

    # --- Resolve constructor owner (public entry point) ---

    def normalize_constructor_ref(self, ref: ConstructorRef) -> ConstructorRef:
        """Use generic metadata from the resolved owner identity.

        Scope records inline-member parameters from syntax so it can publish
        constructor candidates before types are built. Transparent aliases can
        erase such a parameter, however, so the completed record template is
        authoritative once checking begins.
        """
        typedef = self._ctx._env.type_table.get_by_id(ref.owner_decl_node_id)
        if typedef is not None and typedef.kind in {"record", "enum"}:
            return replace(ref, type_params=typedef.type_params)
        return ref

    # --- Constructor as value (public entry point) ---

    def check_constructor_as_value(
        self,
        *,
        owner: RecordType | ExceptionType,
        span: SourceSpan,
        expected: Type | None,
    ) -> Type:
        """Type a non-generic constructor used in value position (not directly called).

        A constructor with fields becomes a ``FunctionType`` (field types →
        owner type) so it can be passed around and called positionally.  A
        zero-field record or nullary variant keeps its bare nominal value (a
        zero-arg construction).  An exception constructor is rejected — its
        construction has special trace-id semantics and is out of scope as a
        first-class value.
        """
        owner = self._ctx._zonk_constructor_owner(owner)
        if isinstance(owner, ExceptionType):
            raise AglTypeError(
                "Exception constructors cannot be used as a first-class value; "
                "construct the exception directly (e.g. `Abort(message: ...)`).",
                span=span,
            )
        self._reject_session_constructor(owner, span)
        fields = self._ctx._env.type_table.record_fields(owner)
        if fields:
            params = tuple(fields.values())
            return self._contextualize_member_result(
                FunctionType(params=params, result=owner), expected, span, owner.name
            )
        return self._check_constructor_call(owner=owner, positional=(), named=(), span=span)

    # --- Constructor callee calls (public entry points) ---

    def check_constructor_callee_call(
        self,
        node: Call,
        *,
        ctor_ref: ConstructorRef,
        sig: ConstructorSignature,
        expected: Type | None,
        hole_indices: Mapping[int, int],
    ) -> Type:
        """Handle a Call whose callee is a constructor VarRef.

        *ctor_ref* is scope's reference, whose declared parameters explicit
        type arguments apply to; *sig* is its signature, quantified over the
        parameters the constructed type mentions.
        """
        if node.type_args:
            type_params = self._explicit_constructor_type_params(
                ctor_ref, node.type_args, node.span
            )
            return self._check_generic_constructor_call(
                node_type_args=node.type_args,
                ctor_ref=replace(ctor_ref, type_params=type_params),
                node=node,
                expected=expected,
                hole_indices=hole_indices,
                sig=sig,
            )
        if sig.type_params:
            return self._check_generic_constructor_call(
                node_type_args=(),
                ctor_ref=replace(ctor_ref, type_params=sig.type_params),
                node=node,
                expected=expected,
                hole_indices=hole_indices,
                sig=sig,
            )
        owner = sig.result_template
        self._reject_session_constructor(owner, node.span)
        self._reject_abstract_exception_constructor(owner, node.span)
        return self._check_constructor_call(
            owner=owner,
            positional=node.args,
            named=node.named_args,
            span=node.span,
            node=node,
            hole_indices=hole_indices,
        )

    # --- Constructor call validation (private helper) ---

    def _reject_session_constructor(
        self, owner: RecordType | ExceptionType, span: SourceSpan
    ) -> None:
        if (
            isinstance(owner, RecordType)
            and owner.decl_id in self._ctx._env.type_table.host_minted_declaration_ids()
        ):
            raise AglTypeError(
                "'Session' values are created by the host and cannot be constructed in source.",
                span=span,
            )

    def _reject_abstract_exception_constructor(
        self, owner: RecordType | EnumType | ExceptionType, span: SourceSpan
    ) -> None:
        owner = self._ctx._zonk_constructor_owner(owner)
        if (
            isinstance(owner, ExceptionType)
            and self._ctx._env.type_table.exception_def(owner).abstract
        ):
            raise AglTypeError(
                "The abstract 'Exception' base type is not constructible. "
                "Use a concrete exception type (e.g. 'Abort').",
                span=span,
            )

    def _check_constructor_call(
        self,
        *,
        owner: RecordType | ExceptionType,
        positional: tuple[CallArg, ...],
        named: tuple[NamedArg[CallArg], ...],
        span: SourceSpan,
        node: Call | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        owner = self._ctx._zonk_constructor_owner(owner)
        fields, context_desc = self._constructor_fields_and_context(owner)

        field_kinds = self._ctx._env.type_table.field_kinds(owner)

        # Bind positional and named args to field names via the shared helper.
        # All fields are required (no defaults on constructors), so every slot is
        # non-None after binding — the helper asserts this internally.
        bound_exprs = bind_constructor_args(
            field_kinds, positional, named, call_span=span, context_desc=context_desc
        )
        return self._finish_constructor_call(
            owner=owner,
            field_kinds=field_kinds,
            bound_exprs=bound_exprs,
            node=node,
            hole_indices={} if hole_indices is None else hole_indices,
        )
