"""Constructor (record/enum-variant/exception, generic + cross-module) call/value checker.

Driven by ``_Checker`` via the narrow ``ConstructorCheckCtx`` Protocol.  All
logic lives here; the host checker instantiates ``ConstructorChecker(self)``
and delegates the constructor dispatch branches in ``_check_varref``,
``_check_varref`` and ``_check_call`` to the public entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, Protocol

from agm.agl.scope.symbols import BindingRef, ConstructorRef, ModuleResolution, ScopePath
from agm.agl.semantics.types import (
    EnumType,
    ExceptionType,
    FunctionType,
    RecordType,
    Type,
    TypeTemplate,
    substitute,
)
from agm.agl.syntax.nodes import Call, Expr, NamedArg, ParamKind, Placeholder, VarRef
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.arguments import bind_constructor_args
from agm.agl.typecheck.env import (
    AglTypeError,
    ConstructorSignature,
    GenericTypeDef,
    TypeEnvironment,
)
from agm.agl.typecheck.inference import ConstraintRole, InferenceEngine

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
        binding: tuple[Expr | None, ...],
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

    def _zonk_constructor_owner(
        self, owner: RecordType | EnumType | ExceptionType
    ) -> RecordType | EnumType | ExceptionType: ...

    def _active_inference_engine(self) -> InferenceEngine: ...

    def _set_generic_constructor_result_provenance(
        self,
        node_id: int,
        result_template: Type,
        field_templates: Mapping[str, Type],
        bound_exprs: Mapping[str, Expr],
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

    @staticmethod
    def _alias_constructor_signature(
        *,
        target_gdef: GenericTypeDef,
        source: TypeTemplate,
        base_sig: ConstructorSignature,
        owner_name: str,
        variant: str | None,
    ) -> ConstructorSignature:
        """Re-express a constructor signature seen through a transparent alias.

        A local/imported type alias is a transparent spelling of an underlying
        generic nominal. Match the alias template against that nominal's
        template so the underlying field templates are rewritten into the
        alias's type parameters, and expose the alias template as the result.
        """
        target_match = TypeTemplate(target_gdef.template, target_gdef.type_params).match(
            source.template
        )
        assert target_match is not None
        target_subst = dict(target_match.bindings)
        return ConstructorSignature(
            owner_name=owner_name,
            variant=variant,
            field_names=base_sig.field_names,
            field_templates=tuple(
                substitute(field, target_subst) for field in base_sig.field_templates
            ),
            result_template=source.template,
            type_params=source.type_params,
        )

    # --- Generic constructor as value ---

    def check_generic_constructor_as_value(
        self,
        *,
        ctor_ref: ConstructorRef,
        span: SourceSpan,
        expected: Type | None,
        sig: ConstructorSignature | None = None,
        gdef: GenericTypeDef | None = None,
        source_name: str | None = None,
    ) -> Type:
        """Handle a generic constructor used as a bare value (not in direct call position).

        For nullary variants (no fields): instantiate from the expected nominal type.
        For payload constructors: instantiate to a FunctionType from expected FunctionType.
        """
        if sig is None:
            ctor_ref, sig, _resolved_gdef = self._generic_constructor_data(ctor_ref)
        assert sig is not None

        return self._ctx._instantiate_generic_constructor_value(
            type_params=ctor_ref.type_params,
            field_templates=sig.field_templates,
            result_template=sig.result_template,
            span=span,
            expected=expected,
            subject=ctor_ref.owner_name,
        )

    # --- Generic constructor type-apply as value (explicit type args) ---

    def _generic_constructor_data(
        self, ctor_ref: ConstructorRef
    ) -> tuple[ConstructorRef, ConstructorSignature, GenericTypeDef | None]:
        """Return generic constructor metadata for a resolved owner identity.

        Scope owns the source spelling and module identity. This helper follows
        a transparent alias only to obtain its already-registered signature;
        it never performs name resolution.
        """
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        gdef = self._ctx._env.get_generic_type_from_module(
            ctor_ref.owner_module_id, owner_name, scope_path=ctor_ref.owner_path
        )
        if gdef is not None:
            sig = self._ctx._env.get_ctor_sig_from_module(
                ctor_ref.owner_module_id, owner_name, variant, scope_path=ctor_ref.owner_path
            )
            if sig is None:
                sig = self._ctx._env.get_constructor_signature(
                    owner_name, variant, scope_path=ctor_ref.owner_path
                )
            assert sig is not None, f"No constructor signature for {owner_name}.{variant}"
            return ctor_ref, sig, gdef

        source = self._ctx._env.source_type_template_qname(
            ctor_ref.owner_module_id, owner_name, scope_path=ctor_ref.owner_path
        )
        if source is not None and isinstance(source.template, (RecordType, EnumType)):
            target = source.template
            target_gdef = self._ctx._env.get_generic_type_from_module(
                target.module_id, target.name, scope_path=target.scope_path
            )
            if target_gdef is None:
                target_gdef = self._ctx._env.get_generic_type(target.name)
            if target_gdef is not None:
                target_sig = self._ctx._env.get_ctor_sig_from_module(
                    target.module_id, target.name, variant, scope_path=target.scope_path
                )
                if target_sig is None:
                    target_sig = self._ctx._env.get_constructor_signature(
                        target.name, variant, scope_path=target.scope_path
                    )
                assert target_sig is not None, (
                    f"No constructor signature for {owner_name}.{variant}"
                )
                return (
                    replace(ctor_ref, type_params=source.type_params),
                    self._alias_constructor_signature(
                        target_gdef=target_gdef,
                        source=source,
                        base_sig=target_sig,
                        owner_name=owner_name,
                        variant=variant,
                    ),
                    target_gdef,
                )

        sig = self._ctx._env.get_constructor_signature(
            owner_name, variant, scope_path=ctor_ref.owner_path
        )
        assert sig is not None, f"No constructor signature for {owner_name}.{variant}"
        return ctor_ref, sig, None

    def _instantiate_constructor_value(
        self,
        *,
        owner_name: str,
        variant: str | None,
        type_params: tuple[str, ...],
        type_args: tuple[TypeExpr, ...],
        sig: ConstructorSignature,
        gdef: GenericTypeDef | None,
        source_name: str,
        span: SourceSpan,
    ) -> Type:
        """Instantiate a generic constructor value from explicit type arguments.

        Shared core of the bare and qualified type-apply-as-value paths. A
        payload variant yields a ``FunctionType`` (field types → owner type);
        a nullary variant yields the constructed nominal value. *gdef* is the
        owning ``GenericTypeDef`` (own-module or cross-module); when ``None``
        the own-module nominal registry is used via *owner_name*.
        """
        if len(type_args) != len(type_params):
            raise AglTypeError(
                f"'{owner_name}' requires {len(type_params)} type argument(s), "
                f"but {len(type_args)} were supplied.",
                span=span,
            )
        subst = {
            p: self._ctx._env.resolve_type_expr(
                ta, span=span, type_vars=self._ctx._current_type_vars
            )
            for p, ta in zip(type_params, type_args)
        }
        if not sig.field_names:
            # Nullary variant: instantiate the nominal type and construct it.
            concrete_type = substitute(sig.result_template, subst)
            assert isinstance(concrete_type, (RecordType, EnumType, ExceptionType))
            return self._check_constructor_call(
                owner=concrete_type, variant=variant, positional=(), named=(), span=span
            )
        concrete_params = tuple(substitute(ft, subst) for ft in sig.field_templates)
        concrete_result = substitute(sig.result_template, subst)
        return FunctionType(params=concrete_params, result=concrete_result)

    def check_constructor_type_apply(
        self,
        *,
        ctor_ref: ConstructorRef,
        type_args: tuple[TypeExpr, ...],
        span: SourceSpan,
    ) -> Type:
        """Type a generic constructor with explicit type args used as a value.

        ``some::[int]``  → ``FunctionType((int,), Option[int])`` (payload variant).
        ``none::[int]``  → the constructed ``Option[int]`` value (nullary variant).
        """
        if not ctor_ref.type_params:
            raise AglTypeError(
                f"'{ctor_ref.owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        ctor_ref, sig, gdef = self._generic_constructor_data(ctor_ref)
        return self._instantiate_constructor_value(
            owner_name=ctor_ref.owner_name,
            variant=ctor_ref.variant,
            type_params=ctor_ref.type_params,
            type_args=type_args,
            sig=sig,
            gdef=gdef,
            source_name=ctor_ref.owner_name,
            span=span,
        )

    # --- Generic constructor call (private helper) ---

    def _generic_constructor_field_kinds(
        self,
        *,
        owner_name: str,
        variant: str | None,
        gdef: GenericTypeDef | None,
        owner_path: ScopePath = (),
    ) -> tuple[tuple[str, ParamKind], ...]:
        field_kinds = (
            self._ctx._env.get_constructor_field_kinds_for_type(gdef.template, owner_name, variant)
            if gdef is not None
            else self._ctx._env.get_constructor_field_kinds(
                owner_name, variant, scope_path=owner_path
            )
        )
        assert field_kinds is not None, (
            f"compiler bug: no field-kinds for generic constructor '{owner_name}'"
        )
        return field_kinds

    def _check_generic_constructor_call(
        self,
        *,
        node_type_args: tuple[TypeExpr, ...],
        ctor_ref: ConstructorRef,
        positional: tuple[Expr, ...],
        named: tuple[NamedArg, ...],
        span: SourceSpan,
        node: Call,
        expected: Type | None,
        hole_indices: Mapping[int, int],
        sig: ConstructorSignature,
        gdef: GenericTypeDef | None = None,
    ) -> Type:
        """Check a generic constructor through the expression-region solver."""
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        type_params = ctor_ref.type_params
        field_kinds = self._generic_constructor_field_kinds(
            owner_name=owner_name, variant=variant, gdef=gdef, owner_path=ctor_ref.owner_path
        )
        bound_exprs = bind_constructor_args(
            field_kinds,
            positional,
            named,
            call_span=span,
            context_desc=f"constructor '{owner_name}'",
        )

        if node_type_args:
            if len(node_type_args) != len(type_params):
                raise AglTypeError(
                    f"'{owner_name}' requires {len(type_params)} type argument(s), "
                    f"but {len(node_type_args)} were supplied.",
                    span=span,
                )
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
                    subject=owner_name,
                    error_subject=f"constructor '{owner_name}'",
                )
        except AglTypeError as exc:
            framed = self._ctx._frame_generic_constraint_error(exc, participants)
            if framed is exc:
                raise
            raise framed from exc

        assert isinstance(result, (RecordType, EnumType))
        produced = self._constructor_call_result_type(
            field_kinds, fields_by_name, result, bound_exprs, hole_indices
        )
        if expected is not None and not node_type_args:
            engine = self._inference_engine()
            engine.complete_from_context(
                produced,
                expected,
                engine.origin(span, role=ConstraintRole.EXPECTED_RESULT, subject=owner_name),
            )
        self._ctx._record_constructor_call_binding(node.node_id, dict(bound_exprs))
        if hole_indices:
            self._ctx._record_partial_call(
                node,
                tuple(bound_exprs[name] for name, _kind in field_kinds),
                hole_indices,
                callee_kind="constructor",
            )
        if not node_type_args:
            self._ctx._set_generic_constructor_result_provenance(
                node.node_id,
                sig.result_template,
                dict(zip(sig.field_names, sig.field_templates, strict=True)),
                bound_exprs,
            )
        return produced

    def _inference_engine(self) -> InferenceEngine:
        """Return the active shared solver for a generic constructor occurrence."""
        return self._ctx._active_inference_engine()

    # --- Constructor call helpers ---

    def _constructor_fields_and_context(
        self, owner: RecordType | EnumType | ExceptionType, variant: str | None
    ) -> tuple[Mapping[str, Type], str]:
        owner = self._ctx._zonk_constructor_owner(owner)
        if isinstance(owner, EnumType):
            assert variant is not None, "variant is required for EnumType"
            return (
                self._ctx._env.type_table.enum_variants(owner)[variant],
                f"variant '{owner.name}.{variant}'",
            )
        if isinstance(owner, RecordType):
            return self._ctx._env.type_table.record_fields(owner), f"constructor '{owner.name}'"
        return self._ctx._env.type_table.exception_fields(owner), f"exception '{owner.name}'"

    @staticmethod
    def _constructor_call_result_type(
        field_kinds: tuple[tuple[str, ParamKind], ...],
        field_types: Mapping[str, Type],
        result: RecordType | EnumType | ExceptionType,
        bound_exprs: Mapping[str, Expr],
        hole_indices: Mapping[int, int],
    ) -> Type:
        if not hole_indices:
            return result
        hole_types: list[Type | None] = [None] * len(hole_indices)
        for fname, _fkind in field_kinds:
            bound_expr = bound_exprs[fname]
            if isinstance(bound_expr, Placeholder):
                hole_types[hole_indices[bound_expr.node_id]] = field_types[fname]
        assert all(typ is not None for typ in hole_types), (
            "compiler bug: partial constructor hole was not bound to a field"
        )
        return FunctionType(
            params=tuple(typ for typ in hole_types if typ is not None),
            result=result,
        )

    def _finish_constructor_call(
        self,
        *,
        owner: RecordType | EnumType | ExceptionType,
        variant: str | None,
        field_kinds: tuple[tuple[str, ParamKind], ...],
        bound_exprs: Mapping[str, Expr],
        node: Call | None,
        hole_indices: Mapping[int, int],
    ) -> Type:
        owner = self._ctx._zonk_constructor_owner(owner)
        fields, _context_desc = self._constructor_fields_and_context(owner, variant)

        if node is not None:
            self._ctx._record_constructor_call_binding(node.node_id, dict(bound_exprs))
            if hole_indices:
                binding: tuple[Expr | None, ...] = tuple(
                    bound_exprs[fname] for fname, _fkind in field_kinds
                )
                self._ctx._record_partial_call(
                    node,
                    binding,
                    hole_indices,
                    callee_kind="constructor",
                )

        # Type-check each user field (exceptions skip trace_id, which is excluded
        # from field_kinds at registration time). Placeholder fields are checked
        # when the produced function is invoked.
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
        """Attach generic metadata discovered from the resolved owner identity."""
        if ref.type_params:
            return ref
        gdef = self._ctx._env.get_generic_type_from_module(
            ref.owner_module_id, ref.owner_name, scope_path=ref.owner_path
        )
        if gdef is not None:
            return replace(ref, type_params=gdef.type_params)
        source = self._ctx._env.source_type_template_qname(
            ref.owner_module_id, ref.owner_name, scope_path=ref.owner_path
        )
        if source is not None and isinstance(source.template, (RecordType, EnumType)):
            return replace(ref, type_params=source.type_params)
        return ref

    def resolve_constructor_owner(
        self, ref: ConstructorRef, span: SourceSpan
    ) -> RecordType | EnumType | ExceptionType:
        """Resolve the owner type for a constructor ref.

        The whole-program type pre-pass registers every module's own
        declarations into the shared program type table before any body is
        checked, so a resolved ``ConstructorRef`` usually finds its owner
        there. Falls back to the unqualified local registry for cross-module
        types that are open-imported but not registered in the shared table
        — including a host builtin (e.g. an exception like ``Abort``) whose
        constructor candidate is ambiently seeded under ``std/core``'s module
        id even when the standard library is not loaded, so the shared table
        never gained an entry for it. Raises a proper diagnostic, rather than
        returning ``None``, when neither lookup finds a constructible owner.
        """
        owner = self._ctx._env.resolve_constructible_type_by_module_id(
            ref.owner_module_id, ref.owner_name, scope_path=ref.owner_path
        )
        if owner is None:
            candidate = self._ctx._env.get_type(ref.owner_name)
            if not isinstance(candidate, (RecordType, EnumType, ExceptionType)):
                raise AglTypeError(
                    f"'{ref.owner_name}' is not a known constructible type.", span=span
                )
            owner = candidate
        if ref.variant is None:
            return owner
        if not isinstance(owner, EnumType):
            if isinstance(owner, (RecordType, ExceptionType)) and ref.variant == ref.owner_name:
                return owner
            raise AglTypeError(f"'{ref.owner_name}' is not a known enum type.", span=span)
        if ref.variant not in self._ctx._env.type_table.enum_variants(owner):
            raise AglTypeError(
                f"Variant '{ref.variant}' does not exist in enum '{ref.owner_name}'.", span=span
            )
        return owner

    # --- Constructor as value (public entry point) ---

    def check_constructor_as_value(
        self,
        *,
        owner: RecordType | EnumType | ExceptionType,
        variant: str | None,
        span: SourceSpan,
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
        if isinstance(owner, EnumType):
            assert variant is not None, "variant is required for EnumType"
            fields = self._ctx._env.type_table.enum_variants(owner)[variant]
        else:
            fields = self._ctx._env.type_table.record_fields(owner)
        if fields:
            params = tuple(fields.values())
            return FunctionType(params=params, result=owner)
        return self._check_constructor_call(
            owner=owner, variant=variant, positional=(), named=(), span=span
        )

    # --- Cross-module constructor value/call (public entry points) ---

    def _resolve_cross_module_nominal_constructor(
        self, callee_ref: BindingRef, span: SourceSpan
    ) -> tuple[RecordType | EnumType | ExceptionType, GenericTypeDef | None]:
        """Resolve a tentative cross-module constructor binding to its nominal target."""
        owner = self._ctx._env.resolve_constructible_type_by_module_id(
            callee_ref.module_id, callee_ref.name, scope_path=callee_ref.scope_path
        )
        if owner is None:
            raise AglTypeError(
                f"'{callee_ref.name}' is a type name, not a constructible nominal type.",
                span=span,
            )
        return owner, self._ctx._env.get_generic_type_from_module(
            owner.module_id, owner.name, scope_path=owner.scope_path
        )

    def _resolve_cross_module_generic_constructor(
        self, callee_ref: BindingRef, span: SourceSpan
    ) -> tuple[
        RecordType | EnumType | ExceptionType, GenericTypeDef, ConstructorSignature, tuple[str, ...]
    ]:
        """Resolve a generic constructor while retaining its source alias template."""
        owner, target_gdef = self._resolve_cross_module_nominal_constructor(callee_ref, span)
        assert target_gdef is not None
        signature = self._ctx._env.get_ctor_sig_from_module(
            owner.module_id, owner.name, None, scope_path=owner.scope_path
        )
        assert signature is not None
        source = self._ctx._env.source_type_template_qname(
            callee_ref.module_id, callee_ref.name, scope_path=callee_ref.scope_path
        )
        assert source is not None
        effective_signature = self._alias_constructor_signature(
            target_gdef=target_gdef,
            source=source,
            base_sig=signature,
            owner_name=callee_ref.name,
            variant=None,
        )
        return owner, target_gdef, effective_signature, source.type_params

    def check_cross_module_constructor_as_value(
        self, callee_ref: BindingRef, *, span: SourceSpan, expected: Type | None
    ) -> Type:
        """Type a module-qualified record constructor used as a value."""
        gdef = self._ctx._env.get_generic_type_from_module(
            callee_ref.module_id, callee_ref.name, scope_path=callee_ref.scope_path
        )
        if gdef is not None:
            if not isinstance(gdef.template, RecordType):
                raise AglTypeError(
                    f"'{callee_ref.name}' is a type name, not a value; "
                    "use it with a constructor call "
                    "(e.g. 'EnumName::Variant' or 'RecordName(...)').",
                    span=span,
                )
            sig = self._ctx._env.get_ctor_sig_from_module(
                callee_ref.module_id, callee_ref.name, None, scope_path=gdef.template.scope_path
            )
            assert sig is not None, (
                f"GenericTypeDef '{callee_ref.name}' in '{callee_ref.module_id.display()}' "
                "has no constructor signature in the program table"
            )
            return self.check_generic_constructor_as_value(
                ctor_ref=ConstructorRef(
                    owner_name=callee_ref.name,
                    variant=None,
                    owner_decl_node_id=callee_ref.decl_node_id,
                    type_params=gdef.type_params,
                ),
                span=span,
                expected=expected,
                sig=sig,
                gdef=gdef,
                source_name=callee_ref.name,
            )
        owner, target_gdef = self._resolve_cross_module_nominal_constructor(callee_ref, span)
        if target_gdef is not None:
            _, target_gdef, sig, type_params = self._resolve_cross_module_generic_constructor(
                callee_ref, span
            )
            return self.check_generic_constructor_as_value(
                ctor_ref=ConstructorRef(
                    owner_name=callee_ref.name,
                    variant=None,
                    owner_decl_node_id=callee_ref.decl_node_id,
                    type_params=type_params,
                ),
                span=span,
                expected=expected,
                sig=sig,
                gdef=target_gdef,
                source_name=callee_ref.name,
            )
        if isinstance(owner, RecordType):
            return self.check_constructor_as_value(owner=owner, variant=None, span=span)
        raise AglTypeError(
            f"'{callee_ref.name}' is a type name, not a value; "
            "use it with a constructor call (e.g. 'EnumName::Variant' or 'RecordName(...)').",
            span=span,
        )

    def check_cross_module_constructor_type_apply(
        self,
        callee_ref: BindingRef,
        *,
        type_args: tuple[TypeExpr, ...],
        span: SourceSpan,
    ) -> Type:
        """Instantiate a module-qualified generic record constructor value."""
        owner, gdef = self._resolve_cross_module_nominal_constructor(callee_ref, span)
        if gdef is None or not isinstance(gdef.template, RecordType):
            raise AglTypeError(
                f"'{callee_ref.name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        _, gdef, sig, type_params = self._resolve_cross_module_generic_constructor(callee_ref, span)
        return self._instantiate_constructor_value(
            owner_name=callee_ref.name,
            variant=None,
            type_params=type_params,
            type_args=type_args,
            sig=sig,
            gdef=gdef,
            source_name=callee_ref.name,
            span=span,
        )

    def check_cross_module_constructor_call(
        self,
        node: Call,
        callee_ref: BindingRef,
        *,
        expected: Type | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        """Handle a Call whose callee is a cross-module constructor VarRef.

        Used when the callee is a qualified VarRef like ``modA::Foo`` that
        resolved to a ``constructor_binding`` in a non-entry module.
        """
        assert isinstance(node.callee, VarRef)
        owner_type, gdef = self._resolve_cross_module_nominal_constructor(callee_ref, node.span)
        if gdef is not None:
            _, gdef, ctor_sig, type_params = self._resolve_cross_module_generic_constructor(
                callee_ref, node.span
            )
            return self._check_generic_constructor_call(
                node_type_args=node.type_args,
                ctor_ref=ConstructorRef(
                    owner_name=callee_ref.name,
                    variant=None,
                    owner_decl_node_id=callee_ref.decl_node_id,
                    type_params=type_params,
                ),
                positional=node.args,
                named=node.named_args,
                span=node.span,
                node=node,
                expected=expected,
                hole_indices={} if hole_indices is None else hole_indices,
                sig=ctor_sig,
                gdef=gdef,
            )
        if node.type_args:
            raise AglTypeError(
                f"'{callee_ref.name}' is not a generic type and does not accept type arguments.",
                span=node.span,
            )
        if isinstance(owner_type, EnumType):
            raise AglTypeError(
                f"'{callee_ref.name}' is an enum type, not a record constructor.",
                span=node.span,
            )
        self._reject_abstract_exception_constructor(owner_type, node.span)
        return self._check_constructor_call(
            owner=owner_type,
            variant=None,
            positional=node.args,
            named=node.named_args,
            span=node.span,
            node=node,
            hole_indices=hole_indices,
        )

    # --- Unqualified constructor callee call (public entry point) ---

    def check_constructor_callee_call(
        self,
        node: Call,
        *,
        ctor_ref: ConstructorRef,
        constructor_type_args: tuple[TypeExpr, ...] | None = None,
        expected: Type | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        """Handle a Call whose callee is an unqualified constructor VarRef."""
        assert isinstance(node.callee, VarRef)
        type_args = () if constructor_type_args is None else constructor_type_args
        if ctor_ref.type_params:
            ctor_ref, sig, gdef = self._generic_constructor_data(ctor_ref)
            return self._check_generic_constructor_call(
                node_type_args=type_args,
                ctor_ref=ctor_ref,
                positional=node.args,
                named=node.named_args,
                span=node.span,
                node=node,
                expected=expected,
                hole_indices={} if hole_indices is None else hole_indices,
                sig=sig,
                gdef=gdef,
            )
        if type_args:
            raise AglTypeError(
                f"'{ctor_ref.owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=node.span,
            )
        owner = self.resolve_constructor_owner(ctor_ref, node.span)
        self._reject_abstract_exception_constructor(owner, node.span)
        return self._check_constructor_call(
            owner=owner,
            variant=ctor_ref.variant,
            positional=node.args,
            named=node.named_args,
            span=node.span,
            node=node,
            hole_indices=hole_indices,
        )

    # --- Constructor call validation (private helper) ---

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
        owner: RecordType | EnumType | ExceptionType,
        variant: str | None,
        positional: tuple[Expr, ...],
        named: tuple[NamedArg, ...],
        span: SourceSpan,
        node: Call | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        owner = self._ctx._zonk_constructor_owner(owner)
        fields, context_desc = self._constructor_fields_and_context(owner, variant)

        # Get field kinds (excludes trace_id for exceptions). The env helper
        # owns the lookup convention (registered table for records/enums,
        # derived from exception_fields for exceptions).
        field_kinds = self._ctx._env.get_constructor_field_kinds_for_type(
            owner, owner.name, variant
        )
        assert field_kinds is not None, (
            f"compiler bug: no field-kinds registered for {context_desc}"
        )

        # Bind positional and named args to field names via the shared helper.
        # All fields are required (no defaults on constructors), so every slot is
        # non-None after binding — the helper asserts this internally.
        bound_exprs = bind_constructor_args(
            field_kinds, positional, named, call_span=span, context_desc=context_desc
        )
        return self._finish_constructor_call(
            owner=owner,
            variant=variant,
            field_kinds=field_kinds,
            bound_exprs=bound_exprs,
            node=node,
            hole_indices={} if hole_indices is None else hole_indices,
        )
