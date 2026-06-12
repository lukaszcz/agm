"""Type-checking pass (Component 5) — M1 implementation.

``check(resolved, capabilities)`` performs a bidirectional type pass over the
``ResolvedProgram``, using the ``HostCapabilities`` to validate codec and
renderer names, and returns a ``CheckedProgram``.

M1 rules implemented
---------------------
1.  Type declaration validation: duplicate names, unknown referenced types,
    recursive records/enums, alias cycles, and built-in-name shadowing.
2.  Binding type inference:
    - ``let/var name: T = e`` — check ``e`` against ``T``.
    - Untyped agent-call binding defaults to ``text`` (design §2.4).
    - Other untyped initializers infer from the literal/expression.
    - ``input name[: T]`` — defaults to ``text`` when unannotated.
3.  ``set name = e`` — expected type is the binding's declared type.
4.  ``print expr`` — accepts any type.
5.  Agent call target typing (§11.4): from annotation / set-target / else
    ``text``.  M1 capability check: target type must be in the ``"text"``
    codec's supported kinds.  Non-text targets are a static error in M1
    ("no registered codec supports type T").
6.  ``strict_json`` is valid only when the selected codec is ``"json"``; in
    M1, using ``strict_json: true`` on a non-JSON codec is always an error.
7.  Renderer names in interpolation segments must exist in capabilities.
8.  Agent names: when ``has_fallback_agent`` is ``False`` an unknown name is
    an error.
9.  Assignability: ``null`` is ``json`` — not assignable to text/int/decimal.
    Int literals are assignable to ``decimal`` (single coercion).
10. Duplicate call options are already rejected by the AST builder; checked
    here for robustness.
11. Type declarations: duplicate fields, duplicate variants, duplicate
    constructor args, duplicate dict keys.

The checker raises ``AglTypeError`` on the first error (Q4 first-error abort).
"""

from __future__ import annotations

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.scope.symbols import BindingRef, CallKind, ResolvedProgram
from agm.agl.syntax.nodes import (
    AgentCall,
    BinaryOp,
    BinOp,
    BoolLit,
    CaseExpr,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictLit,
    DoUntil,
    ElseSentinel,
    EnumDef,
    Expr,
    ExprStmt,
    FieldAccess,
    FieldDef,
    IfStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    LiteralPattern,
    NullLit,
    PrintStmt,
    Program,
    Raise,
    RecordDef,
    SetStmt,
    Stmt,
    StringLit,
    Template,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr
from agm.agl.typecheck.env import AglTypeError, CheckedProgram, OutputContractSpec, TypeEnvironment
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTION_NAMES,
    BoolType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    is_assignable,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in type names that the user may not shadow with a record/enum/alias.
_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {"text", "json", "bool", "int", "decimal"}
) | BUILTIN_EXCEPTION_NAMES


# ---------------------------------------------------------------------------
# Pre-pass: collect and validate type declarations
# ---------------------------------------------------------------------------


class _TypeBuilder:
    """First pass: collect record/enum/alias declarations and validate them.

    Populates a ``TypeEnvironment`` with all user-declared types.  Raises
    ``AglTypeError`` on:
    - Duplicate type names (user vs user, or user shadowing a built-in).
    - Duplicate record fields or enum variants/fields.
    - Unknown type references inside field/variant definitions.
    - Recursive records or enums (v1: rejected).
    - Alias cycles.
    """

    def __init__(self, env: TypeEnvironment) -> None:
        self._env = env
        # Track user-declared names (excludes built-ins) to detect duplicates.
        self._declared: dict[str, SourceSpan] = {}

    def collect(self, program: Program) -> None:
        """Scan *program* and populate ``self._env``."""
        # Pass 1: register all names (forward references permitted between
        # type declarations, consistent with plan §0 "type-decl ordering").
        for stmt in program.body:
            if isinstance(stmt, RecordDef):
                self._register_name(stmt.name, stmt.span)
            elif isinstance(stmt, EnumDef):
                self._register_name(stmt.name, stmt.span)
            elif isinstance(stmt, TypeAlias):
                self._register_name(stmt.name, stmt.span)
                self._env.register_alias(stmt.name, stmt.type_expr)

        # Pass 2: resolve field/variant types and build semantic Type objects.
        for stmt in program.body:
            if isinstance(stmt, RecordDef):
                self._build_record(stmt)
            elif isinstance(stmt, EnumDef):
                self._build_enum(stmt)
            elif isinstance(stmt, TypeAlias):
                # Aliases are resolved lazily via TypeEnvironment.resolve_type_expr;
                # we validate them now to surface cycle/unknown errors early.
                self._validate_alias(stmt)

    def _register_name(self, name: str, span: SourceSpan) -> None:
        if name in _BUILTIN_TYPE_NAMES:
            raise AglTypeError(
                f"'{name}' is a built-in type name and cannot be redeclared.",
                span=span,
            )
        if name in self._declared:
            raise AglTypeError(
                f"Type '{name}' is already declared.",
                span=span,
            )
        self._declared[name] = span

    def _build_record(self, stmt: RecordDef) -> None:
        fields: dict[str, Type] = {}
        seen_fields: dict[str, SourceSpan] = {}
        for fd in stmt.fields:
            if fd.name in seen_fields:
                raise AglTypeError(
                    f"Duplicate field '{fd.name}' in record '{stmt.name}'.",
                    span=fd.span,
                )
            seen_fields[fd.name] = fd.span
            field_type = self._resolve_field_type(fd, stmt.name)
            fields[fd.name] = field_type
        # Note: recursive records are naturally rejected by _resolve_field_type
        # raising AglTypeError("Unknown type …") when the self-referencing type
        # is looked up before it has been fully registered.
        rec = RecordType(name=stmt.name, fields=fields)
        self._env.register_type(stmt.name, rec)

    def _build_enum(self, stmt: EnumDef) -> None:
        variants: dict[str, dict[str, Type]] = {}
        seen_variants: dict[str, SourceSpan] = {}
        for vd in stmt.variants:
            if vd.name in seen_variants:
                raise AglTypeError(
                    f"Duplicate variant '{vd.name}' in enum '{stmt.name}'.",
                    span=vd.span,
                )
            seen_variants[vd.name] = vd.span
            vfields: dict[str, Type] = {}
            seen_vfields: dict[str, SourceSpan] = {}
            for fd in vd.fields:
                if fd.name in seen_vfields:
                    raise AglTypeError(
                        f"Duplicate field '{fd.name}' in variant "
                        f"'{stmt.name}.{vd.name}'.",
                        span=fd.span,
                    )
                seen_vfields[fd.name] = fd.span
                vfields[fd.name] = self._resolve_field_type(fd, f"{stmt.name}.{vd.name}")
            variants[vd.name] = vfields
        enum_t = EnumType(name=stmt.name, variants=variants)
        self._env.register_type(stmt.name, enum_t)

    def _resolve_field_type(self, fd: FieldDef, owner: str) -> Type:
        """Resolve a field's TypeExpr to a semantic Type."""
        try:
            return self._env.resolve_type_expr(fd.type_expr, span=fd.span)
        except AglTypeError as exc:
            # Re-raise to propagate; already has the right span.
            raise exc

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles."""
        try:
            self._env.resolve_type_expr(stmt.type_expr, span=stmt.span)
        except AglTypeError:
            raise


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class _Checker:
    """Stateful type-checking visitor.

    Walks the program's statements in order, maintaining a binding-type lookup
    table (``node_id → Type``) populated by ``_TypeBuilder`` for declarations
    and by inline inference for ``let``/``var``.

    Uses expected-type propagation: ``_check_expr(node, expected)`` propagates
    an outer type context into the expression.  When ``expected`` is ``None``
    the expression is inferred bottom-up.
    """

    def __init__(
        self,
        env: TypeEnvironment,
        resolved: ResolvedProgram,
        capabilities: HostCapabilities,
    ) -> None:
        self._env = env
        self._resolved = resolved
        self._caps = capabilities
        self._node_types: dict[int, Type] = {}
        self._contract_specs: dict[int, OutputContractSpec] = {}
        self._warnings: list[Diagnostic] = []

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def check_program(self, program: Program) -> None:
        for stmt in program.body:
            self._check_stmt(stmt)

    def _check_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, LetDecl):
            self._check_let(stmt)
        elif isinstance(stmt, VarDecl):
            self._check_var(stmt)
        elif isinstance(stmt, SetStmt):
            self._check_set(stmt)
        elif isinstance(stmt, InputDecl):
            self._check_input(stmt)
        elif isinstance(stmt, PrintStmt):
            # print accepts any type.
            self._check_expr(stmt.value, expected=None)
        elif isinstance(stmt, (RecordDef, EnumDef, TypeAlias)):
            pass  # Handled by the pre-pass.
        elif isinstance(stmt, DoUntil):
            self._check_do_until(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_if(stmt)
        elif isinstance(stmt, CaseStmt):
            self._check_case_stmt(stmt)
        elif isinstance(stmt, TryCatch):
            self._check_try_catch(stmt)
        elif isinstance(stmt, Raise):
            self._check_raise(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr(stmt.expr, expected=None)
        else:
            pass  # PassStmt — no-op

    def _check_let(self, stmt: LetDecl) -> None:
        ann_type = self._resolve_annotation(stmt.type_ann, stmt.span)
        val_type = self._check_expr(stmt.value, expected=ann_type)
        if ann_type is not None:
            self._assert_assignable(val_type, ann_type, stmt.span)
            declared_type = ann_type
        else:
            declared_type = val_type
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_var(self, stmt: VarDecl) -> None:
        ann_type = self._resolve_annotation(stmt.type_ann, stmt.span)
        val_type = self._check_expr(stmt.value, expected=ann_type)
        if ann_type is not None:
            self._assert_assignable(val_type, ann_type, stmt.span)
            declared_type = ann_type
        else:
            declared_type = val_type
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_set(self, stmt: SetStmt) -> None:
        ref = self._resolved.resolution[stmt.node_id]
        # resolve_binding always returns non-None for a VarDecl binding that
        # has been type-checked; the scope pass and sequential ordering guarantee this.
        target_type = self._require_binding_type(ref)
        val_type = self._check_expr(stmt.value, expected=target_type)
        self._assert_assignable(val_type, target_type, stmt.span)

    def _check_input(self, stmt: InputDecl) -> None:
        if stmt.annotation is not None:
            typ = self._env.resolve_type_expr(stmt.annotation, span=stmt.span)
        else:
            typ = TextType()
        self._env.set_binding_type(stmt.node_id, typ)

    def _check_do_until(self, stmt: DoUntil) -> None:
        for s in stmt.body:
            self._check_stmt(s)
        cond_type = self._check_expr(stmt.condition, expected=None)
        # Condition should be bool; we don't hard-error in M1 (full M3 check).
        _ = cond_type

    def _check_if(self, stmt: IfStmt) -> None:
        for branch in stmt.branches:
            if not isinstance(branch.cond, ElseSentinel):
                self._check_expr(branch.cond, expected=None)
            for s in branch.body:
                self._check_stmt(s)

    def _check_case_stmt(self, stmt: CaseStmt) -> None:
        subj_type = self._check_expr(stmt.subject, expected=None)
        for branch in stmt.branches:
            self._check_case_stmt_branch(branch, subj_type)

    def _check_case_stmt_branch(self, branch: CaseStmtBranch, subj_type: Type) -> None:
        self._bind_pattern_types(branch.pattern, subj_type, branch)
        for s in branch.body:
            self._check_stmt(s)

    def _check_try_catch(self, stmt: TryCatch) -> None:
        for s in stmt.body:
            self._check_stmt(s)
        for clause in stmt.handlers:
            self._check_catch_clause(clause)

    def _check_catch_clause(self, clause: CatchClause) -> None:
        # Determine the caught exception type.
        if clause.exc_type is None or clause.exc_type == "_":
            # Wildcard: bound as abstract Exception.
            from agm.agl.typecheck.types import EXCEPTION_BASE

            exc_type: ExceptionType = EXCEPTION_BASE
        else:
            resolved = self._env.get_type(clause.exc_type)
            if resolved is None or not isinstance(resolved, ExceptionType):
                raise AglTypeError(
                    f"'{clause.exc_type}' is not a known exception type.",
                    span=clause.span,
                )
            exc_type = resolved
        # Set binding type for the binder (if any).
        if clause.binding is not None:
            self._env.set_binding_type(clause.node_id, exc_type)
        for s in clause.body:
            self._check_stmt(s)

    def _check_raise(self, stmt: Raise) -> None:
        exc_type = self._check_expr(stmt.exc, expected=None)
        # Abstract Exception base is not constructible (design §8.1).
        if isinstance(exc_type, ExceptionType) and exc_type.name == "Exception":
            raise AglTypeError(
                "The abstract 'Exception' base type is not constructible. "
                "Use a concrete exception type (e.g. 'Abort').",
                span=stmt.span,
            )

    # ------------------------------------------------------------------
    # Expression type inference
    # ------------------------------------------------------------------

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Infer/check the type of *expr*, recording it in ``_node_types``.

        ``expected`` carries the outer type context (bidirectional typing).
        Returns the inferred (or confirmed) type.
        """
        typ = self._infer_expr(expr, expected=expected)
        # Every Expr union member exposes an integer ``node_id``; record the type.
        self._node_types[expr.node_id] = typ
        return typ

    def _infer_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Bottom-up inference with optional top-down ``expected`` context."""
        from agm.agl.syntax.nodes import (
            AgentCall,
            BinaryOp,
            CaseExpr,
            Constructor,
            DictLit,
            FieldAccess,
            IsTest,
            ListLit,
            Template,
            UnaryNeg,
            VarRef,
        )

        if isinstance(expr, IntLit):
            return IntType()
        if isinstance(expr, DecimalLit):
            return DecimalType()
        if isinstance(expr, BoolLit):
            return BoolType()
        if isinstance(expr, NullLit):
            return JsonType()
        if isinstance(expr, StringLit):
            return TextType()
        if isinstance(expr, Template):
            return self._check_template(expr)
        if isinstance(expr, VarRef):
            return self._check_varref(expr)
        if isinstance(expr, AgentCall):
            return self._check_agent_call(expr, expected=expected)
        if isinstance(expr, BinaryOp):
            return self._check_binary_op(expr)
        if isinstance(expr, UnaryNot):
            self._check_expr(expr.operand, expected=None)
            return BoolType()
        if isinstance(expr, UnaryNeg):
            return self._check_unary_neg(expr)
        if isinstance(expr, IsTest):
            return self._check_is_test(expr)
        if isinstance(expr, CaseExpr):
            return self._check_case_expr(expr, expected=expected)
        if isinstance(expr, FieldAccess):
            return self._check_field_access(expr)
        if isinstance(expr, Constructor):
            return self._check_constructor(expr, expected=expected)
        if isinstance(expr, ListLit):
            return self._check_list_lit(expr, expected=expected)
        # DictLit is the last member of the Expr union; all others are handled above.
        assert isinstance(expr, DictLit), f"Unexpected expression kind: {type(expr).__name__}"
        return self._check_dict_lit(expr, expected=expected)

    # --- Literals ---

    def _check_template(self, node: Template) -> TextType:
        for seg in node.segments:
            if isinstance(seg, InterpSegment):
                seg_type = self._check_expr(seg.expr, expected=None)
                _ = seg_type  # type consumed by renderer check below
                # Validate renderer name if explicit.
                if seg.render is not None and seg.render != "default":
                    if seg.render not in self._caps.renderer_names:
                        raise AglTypeError(
                            f"Unknown renderer '{seg.render}'. "
                            f"Known renderers: {sorted(self._caps.renderer_names)}.",
                            span=seg.span,
                        )
        return TextType()

    # --- VarRef ---

    def _check_varref(self, node: VarRef) -> Type:
        # The scope pass guarantees every VarRef has a resolution entry.
        ref = self._resolved.resolution[node.node_id]
        # The sequential checker guarantees the binding type has been set.
        return self._require_binding_type(ref)

    def _require_binding_type(self, ref: BindingRef) -> Type:
        """Return the resolved type for *ref*, asserting it has been set.

        Every binding referenced from a reachable ``VarRef`` or ``set`` target
        has its type recorded before the reference is checked: ``let``/``var``/
        ``input`` set it eagerly, catch binders and *enum* pattern variables set
        it when their branch is entered, and constructor patterns on non-enum
        subjects are rejected by ``_bind_pattern_types`` before any body runs.
        A ``None`` result therefore signals an internal invariant violation
        rather than a user error.
        """
        typ = self._env.resolve_binding(ref)
        if typ is None:
            raise AssertionError(
                f"Binding {ref!r} has no recorded type; checker invariant violated."
            )
        return typ

    # --- Agent call ---

    def _check_agent_call(self, node: AgentCall, *, expected: Type | None) -> Type:
        kind = self._resolved.call_kinds.get(node.node_id, CallKind.agent)

        # Determine target type from context, defaulting to text (design §2.4).
        if expected is not None:
            target_type: Type = expected
        else:
            target_type = TextType()

        # Select codec: in M1 the only codec is "text" supporting TextType.
        # Codec selection and option validation are structural properties of the
        # call (independent of agent availability), so they are checked before
        # the agent-capability checks below.
        codec_name = self._select_codec(target_type, node.span)

        # Validate strict_json: only valid for JSON codec.
        if node.options.strict_json is not None and codec_name != "json":
            raise AglTypeError(
                f"'strict_json' is only valid when the codec is 'json'; "
                f"the selected codec for this call is '{codec_name}'.",
                span=node.options.span,
            )

        # Validate agent name against capabilities.
        if kind == CallKind.agent:
            if (
                not self._caps.has_fallback_agent
                and node.agent not in self._caps.agent_names
            ):
                raise AglTypeError(
                    f"Unknown agent '{node.agent}'. The host has no fallback agent "
                    "and this name is not registered.",
                    span=node.span,
                )
        elif kind == CallKind.default_agent:
            # A ``prompt`` call needs either a default agent or a fallback agent.
            if not self._caps.has_default_agent and not self._caps.has_fallback_agent:
                raise AglTypeError(
                    "No default agent is configured; the built-in 'prompt' call "
                    "cannot run. Configure a default agent (the CLI wires the "
                    "runner-backed default agent in M5).",
                    span=node.span,
                )

        # Resolve template (renderers validated inside _check_template).
        self._check_template(node.template)

        # Compute effective strict_json for the contract spec.
        effective_strict = node.options.strict_json if codec_name == "json" else None

        spec = OutputContractSpec(
            target_type=target_type,
            codec_name=codec_name,
            strict_json=effective_strict,
        )
        self._contract_specs[node.node_id] = spec

        return target_type

    def _select_codec(self, target_type: Type, span: SourceSpan) -> str:
        """Select the codec name for *target_type* from capabilities.

        Raises ``AglTypeError`` if no registered codec supports the type.
        """
        kind = target_type.kind
        for codec_name, supported_kinds in self._caps.codec_kinds.items():
            if kind in supported_kinds:
                return codec_name
        raise AglTypeError(
            f"No registered codec supports type '{target_type!r}'. "
            f"(Type kind '{kind}' is not handled by any available codec.)",
            span=span,
        )

    # --- Binary operations ---

    def _check_binary_op(self, node: BinaryOp) -> Type:
        left_type = self._check_expr(node.left, expected=None)
        right_type = self._check_expr(node.right, expected=None)

        op = node.op

        if op in (BinOp.AND, BinOp.OR):
            return BoolType()

        if op in (BinOp.EQ, BinOp.NEQ):
            # Allow with int→decimal widening.
            if not (
                is_assignable(left_type, right_type) or is_assignable(right_type, left_type)
            ):
                raise AglTypeError(
                    f"Equality operands must have the same type; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE):
            # Ordering: only on numeric types (int, decimal); bool not allowed.
            if not (
                isinstance(left_type, (IntType, DecimalType))
                and isinstance(right_type, (IntType, DecimalType))
            ):
                raise AglTypeError(
                    f"Ordering operators require numeric operands (int or decimal); "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op == BinOp.IN:
            return self._check_in_op(left_type, right_type, node.span)

        if op == BinOp.ADD:
            return self._check_add(left_type, right_type, node.span)

        if op == BinOp.SUB:
            return self._check_numeric_binop(left_type, right_type, node.span, "-")

        if op == BinOp.MUL:
            return self._check_numeric_binop(left_type, right_type, node.span, "*")

        if op == BinOp.DIV:
            # Division always yields decimal (design §4.3).
            if not (
                isinstance(left_type, (IntType, DecimalType))
                and isinstance(right_type, (IntType, DecimalType))
            ):
                raise AglTypeError(
                    f"'/' requires numeric operands; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return DecimalType()

        raise AglTypeError(  # pragma: no cover
            f"Unknown binary operator: {op!r}",
            span=node.span,
        )

    def _check_add(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # text + text → text (concatenation)
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return TextType()
        # numeric + numeric → int if both int, else decimal
        if isinstance(left_type, (IntType, DecimalType)) and isinstance(
            right_type, (IntType, DecimalType)
        ):
            if isinstance(left_type, IntType) and isinstance(right_type, IntType):
                return IntType()
            return DecimalType()
        raise AglTypeError(
            f"'+' requires both operands to be text or both to be numeric; "
            f"got '{left_type!r}' and '{right_type!r}'.",
            span=span,
        )

    def _check_numeric_binop(
        self, left_type: Type, right_type: Type, span: SourceSpan, op_str: str
    ) -> Type:
        if not (
            isinstance(left_type, (IntType, DecimalType))
            and isinstance(right_type, (IntType, DecimalType))
        ):
            raise AglTypeError(
                f"'{op_str}' requires numeric operands; "
                f"got '{left_type!r}' and '{right_type!r}'.",
                span=span,
            )
        if isinstance(left_type, IntType) and isinstance(right_type, IntType):
            return IntType()
        return DecimalType()

    def _check_in_op(self, left_type: Type, right_type: Type, span: SourceSpan) -> Type:
        # text in text → substring test
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return BoolType()
        # T in list[T] → element test
        if isinstance(right_type, ListType):
            if not is_assignable(left_type, right_type.elem):
                raise AglTypeError(
                    f"'in' element type mismatch: '{left_type!r}' in 'list[{right_type.elem!r}]'.",
                    span=span,
                )
            return BoolType()
        # text in dict[text, V] → key test
        if isinstance(right_type, DictType) and isinstance(left_type, TextType):
            return BoolType()
        raise AglTypeError(
            f"'in' requires (text in text), (T in list[T]), or (text in dict); "
            f"got '{left_type!r}' in '{right_type!r}'.",
            span=span,
        )

    # --- Unary ---

    def _check_unary_neg(self, node: UnaryNeg) -> Type:
        t = self._check_expr(node.operand, expected=None)
        if not isinstance(t, (IntType, DecimalType)):
            raise AglTypeError(
                f"Unary '-' requires a numeric operand; got '{t!r}'.",
                span=node.span,
            )
        return t

    # --- is / is not ---

    def _check_is_test(self, node: IsTest) -> BoolType:
        expr_type = self._check_expr(node.expr, expected=None)
        if not isinstance(expr_type, EnumType):
            raise AglTypeError(
                f"'is' / 'is not' requires an enum-typed left-hand side; "
                f"got '{expr_type!r}'.",
                span=node.span,
            )
        # Check variant membership.
        if node.variant not in expr_type.variants:
            raise AglTypeError(
                f"Variant '{node.variant}' does not belong to enum '{expr_type.name}'.",
                span=node.span,
            )
        return BoolType()

    # --- case expression ---

    def _check_case_expr(self, node: CaseExpr, *, expected: Type | None) -> Type:
        subj_type = self._check_expr(node.subject, expected=None)
        branch_types: list[Type] = []
        for branch in node.branches:
            self._bind_pattern_types(branch.pattern, subj_type, branch)
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)

        if not branch_types:
            return expected if expected is not None else TextType()

        # All branches must have compatible types (int→decimal widening).
        result_type = branch_types[0]
        for bt in branch_types[1:]:
            if bt == result_type:
                continue
            if is_assignable(bt, result_type) or is_assignable(result_type, bt):
                # Widen to decimal when int meets decimal.
                if isinstance(result_type, IntType) and isinstance(bt, DecimalType):
                    result_type = DecimalType()
                # If result_type is already DecimalType and bt is IntType, no change needed.
            else:
                raise AglTypeError(
                    f"Case expression branches have incompatible types: "
                    f"'{result_type!r}' and '{bt!r}'.",
                    span=node.span,
                )
        return result_type

    # --- Field access ---

    def _check_field_access(self, node: FieldAccess) -> Type:
        obj_type = self._check_expr(node.obj, expected=None)
        # For catch binder access (e.g. `e.raw`) — check the exception type's fields.
        if isinstance(obj_type, ExceptionType):
            if node.field not in obj_type.fields:
                raise AglTypeError(
                    f"Exception type '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return obj_type.fields[node.field]
        if isinstance(obj_type, RecordType):
            if node.field not in obj_type.fields:
                raise AglTypeError(
                    f"Record '{obj_type.name}' has no field '{node.field}'.",
                    span=node.span,
                )
            return obj_type.fields[node.field]
        raise AglTypeError(
            f"Field access requires a record or exception value; got '{obj_type!r}'.",
            span=node.span,
        )

    # --- Constructor ---

    def _check_constructor(self, node: Constructor, *, expected: Type | None) -> Type:
        # Resolve the constructor.
        if node.qualifier is not None:
            # Qualified: Enum.Variant
            enum_type = self._env.get_type(node.qualifier)
            if enum_type is None or not isinstance(enum_type, EnumType):
                raise AglTypeError(
                    f"'{node.qualifier}' is not a known enum type.",
                    span=node.span,
                )
            if node.name not in enum_type.variants:
                raise AglTypeError(
                    f"Variant '{node.name}' does not exist in enum '{node.qualifier}'.",
                    span=node.span,
                )
            return self._check_variant_call(node, enum_type, node.name)
        else:
            # Unqualified: look up in all known types.
            return self._check_unqualified_constructor(node, expected=expected)

    def _check_unqualified_constructor(
        self, node: Constructor, *, expected: Type | None
    ) -> Type:
        name = node.name
        # Try records first.
        rec = self._env.get_type(name)
        if isinstance(rec, RecordType):
            return self._check_record_call(node, rec)
        # Try enums (find a unique matching variant).
        candidates: list[tuple[EnumType, str]] = []
        for type_name in self._env.all_declared_type_names():
            t = self._env.get_type(type_name)
            if isinstance(t, EnumType) and name in t.variants:
                candidates.append((t, name))
        # If expected type is an enum, resolve ambiguity.
        if isinstance(expected, EnumType) and (expected, name) in candidates:
            return self._check_variant_call(node, expected, name)
        if len(candidates) == 1:
            enum_t, variant = candidates[0]
            return self._check_variant_call(node, enum_t, variant)
        if len(candidates) > 1:
            enum_names = ", ".join(sorted(et.name for et, _ in candidates))
            raise AglTypeError(
                f"Constructor '{name}' is ambiguous: it appears in multiple enums "
                f"({enum_names}). Use a qualified name (e.g. EnumName.{name}).",
                span=node.span,
            )
        raise AglTypeError(
            f"Unknown constructor '{name}'.",
            span=node.span,
        )

    def _check_record_call(self, node: Constructor, rec: RecordType) -> RecordType:
        provided = {arg.name: arg for arg in node.args}
        # Check duplicates.
        seen_args: set[str] = set()
        for arg in node.args:
            if arg.name in seen_args:
                raise AglTypeError(
                    f"Duplicate argument '{arg.name}' in constructor call.",
                    span=arg.span,
                )
            seen_args.add(arg.name)
        # Check for unknown fields.
        for arg_name in provided:
            if arg_name not in rec.fields:
                raise AglTypeError(
                    f"Record '{rec.name}' has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )
        # Check for missing fields.
        for field_name in rec.fields:
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call for record '{rec.name}'.",
                    span=node.span,
                )
        # Check argument types.
        for arg in node.args:
            expected_field_type = rec.fields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)
        return rec

    def _check_variant_call(
        self, node: Constructor, enum_type: EnumType, variant: str
    ) -> EnumType:
        vfields = enum_type.variants[variant]
        provided = {arg.name: arg for arg in node.args}
        # Check duplicates.
        seen_args: set[str] = set()
        for arg in node.args:
            if arg.name in seen_args:
                raise AglTypeError(
                    f"Duplicate argument '{arg.name}' in constructor call.",
                    span=arg.span,
                )
            seen_args.add(arg.name)
        # Check for unknown fields.
        for arg_name in provided:
            if arg_name not in vfields:
                raise AglTypeError(
                    f"Variant '{enum_type.name}.{variant}' has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )
        # Check for missing fields.
        for field_name in vfields:
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call "
                    f"for variant '{enum_type.name}.{variant}'.",
                    span=node.span,
                )
        # Check argument types.
        for arg in node.args:
            expected_field_type = vfields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)
        return enum_type

    # --- List / Dict literals ---

    def _check_list_lit(self, node: ListLit, *, expected: Type | None) -> Type:
        # Empty list requires annotation.
        if not node.elements:
            elem_type: Type | None = None
            if isinstance(expected, ListType):
                elem_type = expected.elem
            if elem_type is None:
                raise AglTypeError(
                    "Empty list literal requires a type annotation "
                    "(e.g. 'let xs: list[text] = []').",
                    span=node.span,
                )
            return ListType(elem=elem_type)
        # Non-empty: infer from first element, check rest.
        first_type = self._check_expr(node.elements[0], expected=None)
        for elem in node.elements[1:]:
            et = self._check_expr(elem, expected=first_type)
            if not is_assignable(et, first_type) and not is_assignable(first_type, et):
                raise AglTypeError(
                    f"List literal elements have inconsistent types: "
                    f"'{first_type!r}' and '{et!r}'.",
                    span=node.span,
                )
        return ListType(elem=first_type)

    def _check_dict_lit(self, node: DictLit, *, expected: Type | None) -> Type:
        seen_keys: dict[str, SourceSpan] = {}
        val_type: Type | None = None
        for entry in node.entries:
            key = entry.key.value
            if key in seen_keys:
                raise AglTypeError(
                    f"Duplicate key '{key}' in dict literal.",
                    span=entry.span,
                )
            seen_keys[key] = entry.span
            et = self._check_expr(entry.value, expected=val_type)
            if val_type is None:
                val_type = et
        if val_type is None:
            # Empty dict.
            inner: Type | None = None
            if isinstance(expected, DictType):
                inner = expected.value
            if inner is None:
                raise AglTypeError(
                    "Empty dict literal requires a type annotation.",
                    span=node.span,
                )
            return DictType(value=inner)
        return DictType(value=val_type)

    # ------------------------------------------------------------------
    # Pattern binding helpers
    # ------------------------------------------------------------------

    def _bind_pattern_types(self, pattern: object, subj_type: Type, owner: object) -> None:
        """Record binding types for variables introduced by *pattern*."""
        if isinstance(pattern, WildcardPattern):
            pass
        elif isinstance(pattern, LiteralPattern):
            pass
        elif isinstance(pattern, VarPattern):
            # Simple binding captures subject type.
            # All callers (CaseStmtBranch, CaseExprBranch, PatternField) have node_id.
            self._env.set_binding_type(pattern.node_id, subj_type)
        else:
            # The Pattern union is WildcardPattern | LiteralPattern | VarPattern |
            # ConstructorPattern.  The first three are handled above, so this is
            # always ConstructorPattern.
            assert isinstance(pattern, ConstructorPattern), (
                f"Unexpected pattern kind: {type(pattern).__name__}"
            )
            # Constructor patterns match enum variants (design §6.1). A non-enum
            # subject can never match a constructor pattern, so it is a static
            # error — and the resolver has already bound the pattern's field
            # variables, which would otherwise be left untyped.
            if not isinstance(subj_type, EnumType):
                raise AglTypeError(
                    f"Cannot match constructor pattern '{pattern.name}' against "
                    f"non-enum type '{subj_type!r}'.",
                    span=pattern.span,
                )
            # Look up variant field types and bind each sub-pattern.
            variant_name = pattern.name
            vfields = subj_type.variants.get(variant_name, {})
            for pf in pattern.fields:
                if pf.name not in vfields:
                    raise AglTypeError(
                        f"Variant '{variant_name}' has no field '{pf.name}'.",
                        span=pf.span,
                    )
                field_type = vfields[pf.name]
                self._bind_pattern_types(pf.pattern, field_type, pf)

    # ------------------------------------------------------------------
    # Assignability helpers
    # ------------------------------------------------------------------

    def _assert_assignable(self, value_type: Type, target_type: Type, span: SourceSpan) -> None:
        """Raise AglTypeError if value_type is not assignable to target_type."""
        if is_assignable(value_type, target_type):
            return
        raise AglTypeError(
            f"Type mismatch: expected '{target_type!r}', got '{value_type!r}'.",
            span=span,
        )

    # ------------------------------------------------------------------
    # Annotation resolution
    # ------------------------------------------------------------------

    def _resolve_annotation(self, ann: TypeExpr | None, span: SourceSpan) -> Type | None:
        if ann is None:
            return None
        return self._env.resolve_type_expr(ann, span=span)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def result(self, resolved: ResolvedProgram) -> CheckedProgram:
        return CheckedProgram(
            resolved=resolved,
            node_types=self._node_types,
            contract_specs=self._contract_specs,
            warnings=tuple(self._warnings),
            type_env=self._env,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check(resolved: ResolvedProgram, capabilities: HostCapabilities) -> CheckedProgram:
    """Run the full M1 type-checking pass.

    Parameters
    ----------
    resolved:
        Output of the scope resolution pass.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).

    Returns
    -------
    CheckedProgram
        The annotated program with type side tables and contract specs.

    Raises
    ------
    AglTypeError
        On the first static type violation (first-error abort).
    """
    env = TypeEnvironment()
    program = resolved.program

    # Pre-pass: collect and validate all type declarations.
    builder = _TypeBuilder(env)
    builder.collect(program)

    # Main pass: type-check statements and expressions.
    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_program(program)

    return checker.result(resolved)
