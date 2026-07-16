"""Constructor (record/enum-variant/exception, generic + cross-module) call/value checker.

Driven by ``_Checker`` via the narrow ``ConstructorCheckCtx`` Protocol.  All
logic lives here; the host checker instantiates ``ConstructorChecker(self)``
and delegates the constructor dispatch branches in ``_check_varref``,
``_check_varref`` and ``_check_call`` to the public entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import BindingRef, ConstructorRef, ResolvedProgram
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
    _resolved: ResolvedProgram
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

    def _assert_assignable(
        self, value_type: Type, target_type: Type, span: SourceSpan
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
        sig: ConstructorSignature | None = None,
        gdef: GenericTypeDef | None = None,
        source_name: str | None = None,
    ) -> Type:
        """Handle a generic constructor used as a bare value (not in direct call position).

        For nullary variants (no fields): instantiate from the expected nominal type.
        For payload constructors: instantiate to a FunctionType from expected FunctionType.
        """
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        type_params = ctor_ref.type_params
        # Cross-module callers may supply the graph-table signature up front;
        # otherwise resolve it through the constructor's recorded owner module.
        if sig is None:
            imported_gdef = self._ctx._env.get_generic_type_from_module(
                ctor_ref.owner_module_id, owner_name
            )
            if imported_gdef is not None:
                sig = self._ctx._env.get_ctor_sig_from_module(
                    ctor_ref.owner_module_id, owner_name, variant
                )
        if sig is None:
            sig = self._ctx._env.get_constructor_signature(owner_name, variant)
        assert sig is not None, f"No constructor signature for {owner_name}.{variant}"

        return self._ctx._instantiate_generic_constructor_value(
            type_params=type_params,
            field_templates=sig.field_templates,
            result_template=sig.result_template,
            span=span,
            expected=expected,
            subject=owner_name,
        )

    # --- Generic constructor type-apply as value (explicit type args) ---

    def _resolve_constructor_sig(
        self, *, owner_name: str, variant: str | None
    ) -> tuple[ConstructorSignature, "GenericTypeDef | None", str]:
        """Look up a constructor signature, falling back to open-imported generics.

        Returns ``(sig, imported_gdef_or_None, source_name)`` where *source_name*
        is the owner name to use for instantiation (may differ from the callee
        name for open-imported generic enums whose variants travel separately).
        """
        sig = self._ctx._env.get_constructor_signature(owner_name, variant)
        if sig is not None:
            return sig, None, owner_name
        imported = self._ctx._env.get_open_imported_generic_type(owner_name)
        assert imported is not None, (
            f"No constructor signature for {owner_name}.{variant}"
        )
        module_id, source_name, imported_gdef = imported
        sig = self._ctx._env.get_ctor_sig_from_module(module_id, source_name, variant)
        assert sig is not None, f"No constructor signature for {owner_name}.{variant}"
        return sig, imported_gdef, source_name

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
        sig, imported_gdef, source_name = self._resolve_constructor_sig(
            owner_name=ctor_ref.owner_name, variant=ctor_ref.variant
        )
        return self._instantiate_constructor_value(
            owner_name=ctor_ref.owner_name,
            variant=ctor_ref.variant,
            type_params=ctor_ref.type_params,
            type_args=type_args,
            sig=sig,
            gdef=imported_gdef,
            source_name=source_name,
            span=span,
        )

    def check_qualified_constructor_type_apply(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None,
        type_args: tuple[TypeExpr, ...],
        span: SourceSpan,
    ) -> Type:
        """Type a qualified generic constructor with explicit type args as a value.

        e.g. ``Option[int]::some`` or ``Option[int]::none``.
        """
        generic_constructor = self._resolve_qualified_generic_enum_constructor(
            owner_name=owner_name, variant=variant, owner_module_id=owner_module_id
        )
        if generic_constructor is None:
            raise AglTypeError(
                f"'{owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        gdef, sig, type_params = generic_constructor
        return self._instantiate_constructor_value(
            owner_name=owner_name,
            variant=variant,
            type_params=type_params,
            type_args=type_args,
            sig=sig,
            gdef=gdef,
            source_name=owner_name,
            span=span,
        )

    # --- Generic constructor call (private helper) ---

    def _generic_constructor_field_kinds(
        self,
        *,
        owner_name: str,
        variant: str | None,
        gdef: GenericTypeDef | None,
    ) -> tuple[tuple[str, ParamKind], ...]:
        owner_module_id_for_kinds: ModuleId | None = (
            gdef.template.module_id if gdef is not None else None
        )
        actual_name_for_kinds = (
            gdef.template.name
            if gdef is not None and isinstance(gdef.template, (RecordType, EnumType))
            else owner_name
        )
        field_kinds = self._ctx._env.get_constructor_field_kinds(
            actual_name_for_kinds, variant, module_id=owner_module_id_for_kinds
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
        sig: ConstructorSignature | None = None,
        gdef: GenericTypeDef | None = None,
    ) -> Type:
        """Check a generic constructor through the expression-region solver."""
        owner_name = ctor_ref.owner_name
        variant = ctor_ref.variant
        type_params = ctor_ref.type_params
        if sig is None:
            sig = self._ctx._env.get_constructor_signature(owner_name, variant)
        assert sig is not None, (
            f"No constructor signature for {owner_name}.{variant!r}; "
            "scope resolver should have caught unknown variants."
        )
        field_kinds = self._generic_constructor_field_kinds(
            owner_name=owner_name, variant=variant, gdef=gdef
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

        assert isinstance(result, (RecordType, EnumType))
        produced = self._constructor_call_result_type(
            field_kinds, fields_by_name, result, bound_exprs, hole_indices
        )
        if expected is not None and not node_type_args:
            engine = self._inference_engine()
            engine.complete_from_context(
                produced,
                expected,
                engine.origin(
                    span, role=ConstraintRole.EXPECTED_RESULT, subject=owner_name
                ),
            )
        self._ctx._record_constructor_call_binding(node.node_id, dict(bound_exprs))
        if hole_indices:
            self._ctx._record_partial_call(
                node,
                tuple(bound_exprs[name] for name, _kind in field_kinds),
                hole_indices,
                callee_kind="constructor",
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
            self._ctx._assert_assignable(arg_type, expected_field_type, arg_expr.span)

        return self._constructor_call_result_type(
            field_kinds, fields, owner, bound_exprs, hole_indices
        )

    # --- Resolve constructor owner (public entry point) ---

    def resolve_constructor_owner(
        self, ref: ConstructorRef, span: SourceSpan
    ) -> RecordType | EnumType | ExceptionType:
        """Resolve the owner type for a constructor ref.

        Falls back to the unqualified import map for cross-module types that
        are open-imported but not registered in the local environment.
        """
        owner = self._ctx._env.resolve_type_by_module_id(
            ref.owner_module_id, ref.owner_name
        )
        if owner is None:
            owner = self._ctx._env.get_type(ref.owner_name)
        assert isinstance(owner, (RecordType, EnumType, ExceptionType)), (
            f"'{ref.owner_name}' is not a known constructible type."
        )
        return owner

    # --- Qualified enum constructors (public entry points) ---

    def _resolve_qualified_generic_enum_constructor(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None,
    ) -> tuple[GenericTypeDef, ConstructorSignature, tuple[str, ...]] | None:
        """Resolve a generic enum constructor through a declaration or transparent alias."""
        gdef = (
            self._ctx._env.get_generic_type_from_module(owner_module_id, owner_name)
            if owner_module_id is not None
            else self._ctx._env.get_generic_type(owner_name)
        )
        if gdef is not None:
            sig = (
                self._ctx._env.get_ctor_sig_from_module(owner_module_id, owner_name, variant)
                if owner_module_id is not None
                else self._ctx._env.get_constructor_signature(owner_name, variant)
            )
            assert sig is not None, (
                f"Generic enum '{owner_name}' has no constructor signature for '{variant}'"
            )
            return gdef, sig, gdef.type_params

        source = (
            self._ctx._env.source_type_template_qname(owner_module_id, owner_name)
            if owner_module_id is not None
            else self._ctx._env.source_type_template(owner_name)
        )
        if source is None or not isinstance(source.template, EnumType):
            return None
        target = source.template
        target_gdef = self._ctx._env.get_generic_type_from_module(target.module_id, target.name)
        if target_gdef is None:
            local_gdef = self._ctx._env.get_generic_type(target.name)
            if local_gdef is not None and local_gdef.template.module_id == target.module_id:
                target_gdef = local_gdef
        if target_gdef is None:
            return None
        target_sig = self._ctx._env.get_ctor_sig_from_module(target.module_id, target.name, variant)
        if target_sig is None:
            target_sig = self._ctx._env.get_constructor_signature(target.name, variant)
        assert target_sig is not None, (
            f"Generic enum '{target.name}' has no constructor signature for '{variant}'"
        )
        target_match = TypeTemplate(target_gdef.template, target_gdef.type_params).match(
            source.template
        )
        assert target_match is not None
        target_subst = dict(target_match.bindings)
        return (
            target_gdef,
            ConstructorSignature(
                owner_name=owner_name,
                variant=variant,
                field_names=target_sig.field_names,
                field_templates=tuple(
                    substitute(field, target_subst) for field in target_sig.field_templates
                ),
                result_template=source.template,
                type_params=source.type_params,
            ),
            source.type_params,
        )

    def check_qualified_constructor_as_value(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None,
        span: SourceSpan,
        expected: Type | None,
    ) -> Type:
        """Type a qualified constructor (``Owner::variant``) used in value position."""
        generic_constructor = self._resolve_qualified_generic_enum_constructor(
            owner_name=owner_name, variant=variant, owner_module_id=owner_module_id
        )
        if generic_constructor is not None:
            gdef, sig, type_params = generic_constructor
            # owner_decl_node_id is unused on the as-value path (only owner_name,
            # variant, and type_params are consumed); pass the 0 placeholder.
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=type_params,
            )
            return self.check_generic_constructor_as_value(
                ctor_ref=ctor_ref,
                span=span,
                expected=expected,
                sig=sig,
                gdef=gdef if owner_module_id is not None else None,
                source_name=owner_name,
            )
        enum_type = self._resolve_qualified_enum_owner(
            owner_name, variant, span, owner_module_id=owner_module_id
        )
        return self.check_constructor_as_value(
            owner=enum_type, variant=variant, span=span
        )

    # --- Resolve qualified enum owner (private helper) ---

    def _resolve_qualified_enum_owner(
        self,
        owner_name: str,
        variant: str,
        span: SourceSpan,
        *,
        owner_module_id: ModuleId | None = None,
    ) -> EnumType:
        """Resolve a non-generic qualified constructor's owner to a validated enum.

        Scope records ``Owner::member`` for any declared type name without
        checking enum-ness or variant existence, so both are validated here.
        When ``owner_module_id`` is given (cross-module constructor ref), look up
        directly in the graph type table instead of the unqualified import map.
        """
        if owner_module_id is not None:
            enum_type = self._ctx._env.resolve_type_by_module_id(owner_module_id, owner_name)
        else:
            enum_type = self._ctx._env.resolve_named_type(owner_name)
        if not isinstance(enum_type, EnumType):
            raise AglTypeError(
                f"'{owner_name}' is not a known enum type.",
                span=span,
            )
        enum_type = self._ctx._zonk_constructor_owner(enum_type)
        assert isinstance(enum_type, EnumType)
        if variant not in self._ctx._env.type_table.enum_variants(enum_type):
            raise AglTypeError(
                f"Variant '{variant}' does not exist in enum '{owner_name}'.",
                span=span,
            )
        return enum_type

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

    # --- Resolve qualified constructor and call (private helper) ---

    def _resolve_qualified_constructor_and_call(
        self,
        *,
        owner_name: str,
        variant: str,
        owner_module_id: ModuleId | None = None,
        positional: tuple[Expr, ...],
        named: tuple[NamedArg, ...],
        span: SourceSpan,
        node: Call,
        expected: Type | None = None,
        type_args: tuple[TypeExpr, ...] = (),
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        """Validate and dispatch a qualified constructor (EnumName::variant)."""
        generic_constructor = self._resolve_qualified_generic_enum_constructor(
            owner_name=owner_name, variant=variant, owner_module_id=owner_module_id
        )
        if generic_constructor is not None:
            gdef, sig, type_params = generic_constructor
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=type_params,
            )
            return self._check_generic_constructor_call(
                node_type_args=type_args,
                ctor_ref=ctor_ref,
                positional=positional,
                named=named,
                span=span,
                node=node,
                expected=expected,
                hole_indices={} if hole_indices is None else hole_indices,
                sig=sig,
                gdef=gdef,
            )
        if type_args:
            raise AglTypeError(
                f"'{owner_name}::{variant}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        enum_type = self._resolve_qualified_enum_owner(
            owner_name, variant, span, owner_module_id=owner_module_id
        )
        return self._check_constructor_call(
            owner=enum_type, variant=variant, positional=positional, named=named, span=span,
            node=node,
            hole_indices=hole_indices,
        )

    # --- Cross-module constructor value/call (public entry points) ---

    def _resolve_cross_module_nominal_constructor(
        self, callee_ref: BindingRef, span: SourceSpan
    ) -> tuple[RecordType | EnumType | ExceptionType, GenericTypeDef | None]:
        """Resolve a tentative cross-module constructor binding to its nominal target."""
        owner = self._ctx._env.resolve_constructible_type_by_module_id(
            callee_ref.module_id, callee_ref.name
        )
        if owner is None:
            raise AglTypeError(
                f"'{callee_ref.name}' is a type name, not a constructible nominal type.",
                span=span,
            )
        return owner, self._ctx._env.get_generic_type_from_module(owner.module_id, owner.name)

    def _resolve_cross_module_generic_constructor(
        self, callee_ref: BindingRef, span: SourceSpan
    ) -> tuple[
        RecordType | EnumType | ExceptionType, GenericTypeDef, ConstructorSignature, tuple[str, ...]
    ]:
        """Resolve a generic constructor while retaining its source alias template."""
        owner, target_gdef = self._resolve_cross_module_nominal_constructor(callee_ref, span)
        assert target_gdef is not None
        signature = self._ctx._env.get_ctor_sig_from_module(owner.module_id, owner.name, None)
        assert signature is not None
        source = self._ctx._env.source_type_template_qname(callee_ref.module_id, callee_ref.name)
        assert source is not None
        target_match = TypeTemplate(target_gdef.template, target_gdef.type_params).match(
            source.template
        )
        assert target_match is not None
        target_subst = dict(target_match.bindings)
        effective_signature = ConstructorSignature(
            owner_name=callee_ref.name,
            variant=None,
            field_names=signature.field_names,
            field_templates=tuple(
                substitute(field, target_subst) for field in signature.field_templates
            ),
            result_template=source.template,
            type_params=source.type_params,
        )
        return owner, target_gdef, effective_signature, source.type_params

    def check_cross_module_constructor_as_value(
        self, callee_ref: BindingRef, *, span: SourceSpan, expected: Type | None
    ) -> Type:
        """Type a module-qualified record constructor used as a value."""
        gdef = self._ctx._env.get_generic_type_from_module(callee_ref.module_id, callee_ref.name)
        if gdef is not None:
            if not isinstance(gdef.template, RecordType):
                raise AglTypeError(
                    f"'{callee_ref.name}' is a type name, not a value; "
                    "use it with a constructor call "
                    "(e.g. 'EnumName::Variant' or 'RecordName(...)').",
                    span=span,
                )
            sig = self._ctx._env.get_ctor_sig_from_module(
                callee_ref.module_id, callee_ref.name, None
            )
            assert sig is not None, (
                f"GenericTypeDef '{callee_ref.name}' in '{callee_ref.module_id.dotted()}' "
                "has no constructor signature in the graph table"
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
                f"'{callee_ref.name}' is not a generic type and does not accept "
                "type arguments.",
                span=node.span,
            )
        if isinstance(owner_type, EnumType):
            raise AglTypeError(
                f"'{callee_ref.name}' is an enum type, not a record constructor.",
                span=node.span,
            )
        self._reject_abstract_exception_constructor(owner_type, node.span)
        return self._check_constructor_call(
            owner=owner_type, variant=None, positional=node.args, named=node.named_args,
            span=node.span, node=node, hole_indices=hole_indices,
        )

    # --- Unqualified constructor callee call (public entry point) ---

    def check_constructor_callee_call(
        self,
        node: Call,
        *,
        expected: Type | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        """Handle a Call whose callee is an unqualified constructor VarRef."""
        assert isinstance(node.callee, VarRef)
        ctor_ref = self._ctx._resolved.constructor_refs[node.callee.node_id]
        if ctor_ref.type_params:
            # Generic constructor: route to generic call handler.
            gdef = None
            sig = None
            # Look up the imported generic type by its OWNER name, not the callee
            # name: for an enum variant constructor (e.g. `some`), the callee name
            # is the variant, but only the enum TYPE name (`Option`) is registered
            # in the import map (enum variants travel with their enum).
            gdef = self._ctx._env.get_generic_type_from_module(
                ctor_ref.owner_module_id, ctor_ref.owner_name
            )
            if gdef is not None:
                sig = self._ctx._env.get_ctor_sig_from_module(
                    ctor_ref.owner_module_id, ctor_ref.owner_name, ctor_ref.variant
                )
            else:
                source = self._ctx._env.source_type_template_qname(
                    ctor_ref.owner_module_id, ctor_ref.owner_name
                )
                if source is not None and isinstance(source.template, RecordType):
                    target = source.template
                    gdef = self._ctx._env.get_generic_type_from_module(
                        target.module_id, target.name
                    )
                    if gdef is None:
                        gdef = self._ctx._env.get_generic_type(target.name)
                    if gdef is not None:
                        target_sig = self._ctx._env.get_ctor_sig_from_module(
                            target.module_id, target.name, None
                        )
                        if target_sig is None:
                            target_sig = self._ctx._env.get_constructor_signature(target.name, None)
                        assert target_sig is not None
                        target_match = TypeTemplate(gdef.template, gdef.type_params).match(
                            source.template
                        )
                        assert target_match is not None
                        target_subst = dict(target_match.bindings)
                        sig = ConstructorSignature(
                            owner_name=ctor_ref.owner_name,
                            variant=None,
                            field_names=target_sig.field_names,
                            field_templates=tuple(
                                substitute(field, target_subst)
                                for field in target_sig.field_templates
                            ),
                            result_template=source.template,
                            type_params=source.type_params,
                        )
                        ctor_ref = ConstructorRef(
                            owner_name=ctor_ref.owner_name,
                            variant=None,
                            owner_decl_node_id=ctor_ref.owner_decl_node_id,
                            type_params=source.type_params,
                            owner_module_id=ctor_ref.owner_module_id,
                        )
            return self._check_generic_constructor_call(
                node_type_args=node.type_args,
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
        if node.type_args:
            raise AglTypeError(
                f"'{ctor_ref.owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=node.span,
            )
        owner = self.resolve_constructor_owner(ctor_ref, node.span)
        self._reject_abstract_exception_constructor(owner, node.span)
        return self._check_constructor_call(
            owner=owner, variant=ctor_ref.variant, positional=node.args, named=node.named_args,
            span=node.span, node=node, hole_indices=hole_indices,
        )

    # --- Qualified constructor callee call (public entry point) ---

    def check_qualified_constructor_callee_call(
        self,
        node: Call,
        *,
        expected: Type | None = None,
        hole_indices: Mapping[int, int] | None = None,
    ) -> Type:
        """Handle a Call whose callee is a qualified constructor VarRef."""
        assert isinstance(node.callee, VarRef)
        owner_name, variant, owner_module_id = (
            self._ctx._resolved.qualified_constructor_refs[node.callee.node_id]
        )
        type_args: tuple[TypeExpr, ...] = ()
        if (
            node.callee.type_qualifier is not None
            and node.callee.type_qualifier.type_args is not None
        ):
            type_args = node.callee.type_qualifier.type_args
        return self._resolve_qualified_constructor_and_call(
            owner_name=owner_name, variant=variant, owner_module_id=owner_module_id,
            positional=node.args, named=node.named_args, span=node.span,
            node=node,
            expected=expected,
            type_args=type_args,
            hole_indices=hole_indices,
        )

    # --- Constructor call validation (private helper) ---

    def _reject_abstract_exception_constructor(
        self, owner: RecordType | EnumType | ExceptionType, span: SourceSpan
    ) -> None:
        owner = self._ctx._zonk_constructor_owner(owner)
        if isinstance(owner, ExceptionType) and self._ctx._env.type_table.exception_def(
            owner
        ).abstract:
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
