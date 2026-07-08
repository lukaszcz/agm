"""Constructor (record/enum-variant/exception, generic + cross-module) call/value checker.

Driven by ``_Checker`` via the narrow ``ConstructorCheckCtx`` Protocol.  All
logic lives here; the host checker instantiates ``ConstructorChecker(self)``
and delegates the constructor dispatch branches in ``_check_varref``,
``_check_varref`` and ``_check_call`` to the public entry points.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, Protocol

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import BindingRef, ConstructorRef, ResolvedProgram
from agm.agl.semantics.types import (
    EnumType,
    ExceptionType,
    FunctionType,
    RecordType,
    Type,
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

# ---------------------------------------------------------------------------
# Narrow context Protocol
# ---------------------------------------------------------------------------


class ConstructorCheckCtx(Protocol):
    """The minimal _Checker surface the constructor checker needs."""

    _env: TypeEnvironment
    _resolved: ResolvedProgram
    _current_type_vars: frozenset[str]
    _constructor_call_bindings: dict[int, dict[str, Expr]]


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

    def _match(
        self,
        template: Type,
        concrete: Type,
        subst: dict[str, Type],
        *,
        span: SourceSpan,
        challenge: bool = True,
    ) -> None: ...

    def _infer_arg(
        self,
        template: Type,
        arg_expr: Expr,
        subst: dict[str, Type],
        hint: Mapping[str, Type],
        *,
        span: SourceSpan,
    ) -> None: ...

    def _require_all_solved(
        self,
        type_params: tuple[str, ...],
        subst: Mapping[str, Type],
        *,
        span: SourceSpan,
        message_for: Callable[[str], str],
    ) -> None: ...



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
        # Open-imported and cross-module qualified generic constructors used as bare
        # values may supply their graph-table signature up front; otherwise start
        # with the own-module registry and fall back to the open-import map.
        imported_gdef: GenericTypeDef | None = gdef
        imported_source_name = source_name or owner_name
        if sig is None:
            sig = self._ctx._env.get_constructor_signature(owner_name, variant)
        if sig is None:
            # A generic constructor with no own-module signature must be open-imported
            # (the scope resolver guarantees the reference resolved to some type).
            imported = self._ctx._env.get_open_imported_generic_type(owner_name)
            assert imported is not None, (
                f"No constructor signature for {owner_name}.{variant}"
            )
            module_id, imported_source_name, imported_gdef = imported
            sig = self._ctx._env.get_ctor_sig_from_module(
                module_id, imported_source_name, variant
            )
        assert sig is not None, f"No constructor signature for {owner_name}.{variant}"

        if not sig.field_names:
            # Nullary variant: infer type args from the expected nominal enum type.
            # Match on FULL nominal identity (name AND owning module), not just the
            # name — two modules may export same-named generic enums, and borrowing
            # type args from the wrong one would mis-instantiate this constructor.
            if imported_gdef is not None:
                owner_module_id = imported_gdef.template.module_id
                nominal_name = imported_source_name
            else:
                local_gdef = self._ctx._env.get_generic_type(owner_name)
                assert local_gdef is not None, f"No generic type def for '{owner_name}'"
                owner_module_id = local_gdef.template.module_id
                nominal_name = owner_name
            subst: dict[str, Type] = {}
            if (
                expected is not None
                and isinstance(expected, EnumType)
                and expected.name == nominal_name
                and expected.module_id == owner_module_id
            ):
                for p, ta in zip(type_params, expected.type_args):
                    subst[p] = ta
            self._ctx._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument(s) for '{owner_name}': "
                    "no contextual type available. "
                    f"Add a type annotation (e.g. 'let x: {owner_name}[…] = …')."
                ),
            )
            concrete_args = tuple(subst[p] for p in type_params)
            concrete_type = (
                self._ctx._env.instantiate_from_gdef(
                    imported_source_name, imported_gdef, concrete_args
                )
                if imported_gdef is not None
                else self._ctx._env.instantiate_nominal(owner_name, concrete_args)
            )
            return self._check_constructor_call(
                owner=concrete_type, variant=variant, positional=(), named=(), span=span
            )
        else:
            # Payload constructor as value: produce a FunctionType.
            subst = {}
            if expected is not None and isinstance(expected, FunctionType):
                # Match field templates against expected function params.
                for ft, ep in zip(sig.field_templates, expected.params):
                    self._ctx._match(ft, ep, subst, span=span, challenge=False)
                self._ctx._match(
                    sig.result_template, expected.result, subst, span=span, challenge=False
                )
            self._ctx._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument(s) for constructor '{owner_name}': "
                    "no contextual type available. "
                    f"Add a type annotation (e.g. 'let f: ({owner_name}[…]) = …')."
                ),
            )
            concrete_params = tuple(substitute(ft, subst) for ft in sig.field_templates)
            concrete_result = substitute(sig.result_template, subst)
            return FunctionType(params=concrete_params, result=concrete_result)

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
        concrete_args = tuple(subst[p] for p in type_params)
        if not sig.field_names:
            # Nullary variant: instantiate the nominal type and construct it.
            concrete_type = (
                self._ctx._env.instantiate_from_gdef(
                    source_name, gdef, concrete_args, span=span
                )
                if gdef is not None
                else self._ctx._env.instantiate_nominal(owner_name, concrete_args, span=span)
            )
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
        gdef = (
            self._ctx._env.get_generic_type_from_module(owner_module_id, owner_name)
            if owner_module_id is not None
            else self._ctx._env.get_generic_type(owner_name)
        )
        if gdef is None:
            raise AglTypeError(
                f"'{owner_name}' is not a generic constructor and does not accept "
                "type arguments.",
                span=span,
            )
        sig = (
            self._ctx._env.get_ctor_sig_from_module(owner_module_id, owner_name, variant)
            if owner_module_id is not None
            else self._ctx._env.get_constructor_signature(owner_name, variant)
        )
        assert sig is not None, (
            f"Generic enum '{owner_name}' has no constructor signature for '{variant}'"
        )
        return self._instantiate_constructor_value(
            owner_name=owner_name,
            variant=variant,
            type_params=gdef.type_params,
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

    def _seed_generic_constructor_inference(
        self,
        sig: ConstructorSignature,
        templates_by_name: Mapping[str, Type],
        field_kinds: tuple[tuple[str, ParamKind], ...],
        bound_exprs: Mapping[str, Expr],
        hole_indices: Mapping[int, int],
        expected: Type | None,
        subst: dict[str, Type],
        *,
        span: SourceSpan,
    ) -> None:
        if not hole_indices:
            if expected is not None:
                self._ctx._match(
                    sig.result_template, expected, subst, span=span, challenge=False
                )
            return
        if not isinstance(expected, FunctionType) or len(expected.params) != len(hole_indices):
            return
        hole_templates: list[Type | None] = [None] * len(hole_indices)
        for fname, _fkind in field_kinds:
            bound_expr = bound_exprs[fname]
            if isinstance(bound_expr, Placeholder):
                hole_templates[hole_indices[bound_expr.node_id]] = templates_by_name[fname]
        assert all(template is not None for template in hole_templates), (
            "compiler bug: partial constructor hole was not bound to a field"
        )
        for template, concrete in zip(hole_templates, expected.params):
            assert template is not None
            self._ctx._match(template, concrete, subst, span=span, challenge=False)
        self._ctx._match(sig.result_template, expected.result, subst, span=span, challenge=False)

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
        """Check a generic constructor call (with inference or explicit type args).

        ``sig`` and ``gdef`` may be supplied by cross-module callers that already
        looked up these from the graph tables; when ``None``, they are looked up
        from the own-module env (the default path for same-module generic calls).
        """
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

        subst: dict[str, Type] = {}

        if node_type_args:
            # Explicit type arguments path.
            if len(node_type_args) != len(type_params):
                raise AglTypeError(
                    f"'{owner_name}' requires {len(type_params)} type argument(s), "
                    f"but {len(node_type_args)} were supplied.",
                    span=span,
                )
            for p, ta in zip(type_params, node_type_args):
                resolved_arg = self._ctx._env.resolve_type_expr(
                    ta, span=span, type_vars=self._ctx._current_type_vars
                )
                subst[p] = resolved_arg
        else:
            # Inference path: infer type-parameter substitutions from non-hole
            # arguments, using contextual result/produced-function type as a hint.
            templates_by_name = dict(zip(sig.field_names, sig.field_templates))
            hint: dict[str, Type] = {}
            self._seed_generic_constructor_inference(
                sig,
                templates_by_name,
                field_kinds,
                bound_exprs,
                hole_indices,
                expected,
                hint,
                span=span,
            )
            for fname, _fkind in field_kinds:
                bound_expr = bound_exprs[fname]
                if isinstance(bound_expr, Placeholder):
                    continue
                self._ctx._infer_arg(
                    templates_by_name[fname], bound_expr, subst, hint, span=bound_expr.span
                )
            self._seed_generic_constructor_inference(
                sig,
                templates_by_name,
                field_kinds,
                bound_exprs,
                hole_indices,
                expected,
                subst,
                span=span,
            )
            # Verify all type params were solved.
            self._ctx._require_all_solved(
                type_params,
                subst,
                span=span,
                message_for=lambda p: (
                    f"Cannot infer type argument '{p}' for constructor '{owner_name}'; "
                    f"supply it explicitly via '{owner_name}::[…]' or add a type annotation."
                ),
            )

        # Instantiate the nominal type.
        concrete_args = tuple(subst[p] for p in type_params)
        if gdef is not None:
            concrete_type = self._ctx._env.instantiate_from_gdef(
                owner_name, gdef, concrete_args, span=span
            )
        else:
            concrete_type = self._ctx._env.instantiate_nominal(owner_name, concrete_args, span=span)
        assert isinstance(concrete_type, (RecordType, EnumType))
        return self._finish_constructor_call(
            owner=concrete_type,
            variant=variant,
            field_kinds=field_kinds,
            bound_exprs=bound_exprs,
            node=node,
            hole_indices=hole_indices,
        )

    # --- Constructor call helpers ---

    def _constructor_fields_and_context(
        self, owner: RecordType | EnumType | ExceptionType, variant: str | None
    ) -> tuple[Mapping[str, Type], str]:
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
        fields, _context_desc = self._constructor_fields_and_context(owner, variant)

        if node is not None:
            self._ctx._constructor_call_bindings[node.node_id] = dict(bound_exprs)
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
        owner: Type | None = self._ctx._env.get_type(ref.owner_name)
        if owner is None:
            owner = self._ctx._env.resolve_named_type(ref.owner_name)
        assert isinstance(owner, (RecordType, EnumType, ExceptionType)), (
            f"'{ref.owner_name}' is not a known constructible type."
        )
        return owner

    # --- Qualified constructor as value (public entry point) ---

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
            # owner_decl_node_id is unused on the as-value path (only owner_name,
            # variant, and type_params are consumed); pass the 0 placeholder.
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=gdef.type_params,
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
        # Check if this is a generic enum type.
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
            ctor_ref = ConstructorRef(
                owner_name=owner_name,
                variant=variant,
                owner_decl_node_id=0,
                type_params=gdef.type_params,
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

    # --- Cross-module constructor call (public entry point) ---

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
        # Cross-module generic constructor — both explicit (lib::Box[int](v = 1)) and
        # inferred (lib::Box(value = 1)) routes go here.
        gdef = self._ctx._env.get_generic_type_from_module(callee_ref.module_id, callee_ref.name)
        if gdef is not None:
            ctor_sig = self._ctx._env.get_ctor_sig_from_module(
                callee_ref.module_id, callee_ref.name, None
            )
            assert ctor_sig is not None, (
                f"GenericTypeDef '{callee_ref.name}' in '{callee_ref.module_id.dotted()}' "
                "has no constructor signature in the graph table"
            )
            ctor_ref = ConstructorRef(
                owner_name=callee_ref.name,
                variant=None,
                owner_decl_node_id=callee_ref.decl_node_id,
                type_params=gdef.type_params,
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
                sig=ctor_sig,
                gdef=gdef,
            )
        if node.type_args:
            raise AglTypeError(
                f"'{callee_ref.name}' is not a generic type and does not accept "
                "type arguments.",
                span=node.span,
            )
        owner_type = self._ctx._env.resolve_type_by_module_id(callee_ref.module_id, callee_ref.name)
        # The scope resolver sets constructor_binding for RecordDef, EnumDef, and
        # ExceptionDef, all of which map to their respective semantic types in the
        # graph type table.
        assert isinstance(owner_type, (RecordType, EnumType, ExceptionType)), (
            f"constructor_binding for '{callee_ref.name}' in "
            f"'{callee_ref.module_id.dotted()}' resolved to {type(owner_type).__name__}"
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
            imported = self._ctx._env.get_open_imported_generic_type(ctor_ref.owner_name)
            if imported is not None:
                module_id, source_name, gdef = imported
                sig = self._ctx._env.get_ctor_sig_from_module(
                    module_id, source_name, ctor_ref.variant
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
