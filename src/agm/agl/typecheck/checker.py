"""Type-checking pass (Component 5).

``check(resolved, capabilities)`` performs a bidirectional type pass over the
``ResolvedProgram``, using the ``HostCapabilities`` to validate codec and
renderer names, and returns a ``CheckedProgram``.

Rules implemented
-----------------
1.  Type declaration validation: duplicate names, unknown referenced types,
    recursive records/enums, alias cycles, and built-in-name shadowing.
2.  Binding type inference:
    - ``let/var name: T = e`` — check ``e`` against ``T``.
    - Untyped agent-call binding defaults to ``text`` (design §2.4).
    - Other untyped initializers infer from the literal/expression.
    - ``input name[: T]`` — defaults to ``text`` when unannotated.
3.  ``set name = e`` — expected type is the binding's declared type.
4.  ``print expr`` — accepts any type.
5.  Agent-call target typing (§11.4): from annotation / set-target / else
    ``text``.  The target type's kind must be supported by some registered
    codec ("no registered codec supports type T" otherwise).
6.  ``strict_json`` is valid only when the selected codec is ``"json"``.
7.  Renderer names in interpolation segments must exist in capabilities.
8.  Agent names: when ``has_fallback_agent`` is ``False`` an unknown name is
    an error.
9.  Assignability (design §5.8): ``int`` widens to ``decimal``; ``json``
    accepts any JSON-shaped value (scalars and ``list``/``dict`` thereof, but
    not records/enums/exceptions).  List/dict literals propagate the expected
    element/value type and assert every element soundly.  Qualified enum
    constructors, ``is`` tests, and patterns resolve their qualifier
    alias-transparently (§5.4).
10. Duplicate call options are already rejected by the AST builder; checked
    here for robustness.
11. Type declarations: duplicate fields, duplicate variants, duplicate
    constructor args, duplicate dict keys.

The checker raises ``AglTypeError`` on the first error (Q4 first-error abort).
"""

from __future__ import annotations

from collections.abc import Sequence

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.scope.symbols import BindingRef, CallKind, ResolvedProgram
from agm.agl.syntax.nodes import (
    AbortPolicy,
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
    Pattern,
    PrintStmt,
    Program,
    Raise,
    RecordDef,
    RetryPolicy,
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
from agm.agl.typecheck.env import (
    AglTypeError,
    CallSiteRecord,
    CheckedProgram,
    OutputContractSpec,
    TypeEnvironment,
)
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
    comparable_types,
    is_assignable,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in type names that the user may not shadow with a record/enum/alias.
_BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {"text", "json", "bool", "int", "decimal"}
) | BUILTIN_EXCEPTION_NAMES


def _parse_policy_str(policy: AbortPolicy | RetryPolicy | None) -> str:
    """Render a parse policy as the inventory string (``"abort"``/``"retry[N]"``/``"default"``)."""
    if isinstance(policy, AbortPolicy):
        return "abort"
    if isinstance(policy, RetryPolicy):
        return f"retry[{policy.extra}]"
    return "default"


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

    Implementation uses a two-phase approach so that type declarations are
    order-independent (design §0 "type-decl ordering"):

    Phase 1  Register all user-declared type names and alias targets.
             Empty shells (RecordType/EnumType with empty fields/variants)
             are registered immediately so that forward references within the
             same program resolve without "Unknown type" false-positives.
    Phase 2  Resolve every field and variant type, tracking the set of types
             currently under construction (``_building``) to detect direct or
             indirect recursion and report a clear diagnostic.
    """

    def __init__(self, env: TypeEnvironment) -> None:
        self._env = env
        # Track user-declared names → declaration span (excludes built-ins).
        self._declared: dict[str, SourceSpan] = {}
        # Index of record/enum definitions for on-demand phase-2 building.
        self._record_defs: dict[str, RecordDef] = {}
        self._enum_defs: dict[str, EnumDef] = {}
        # Names currently being resolved in phase 2 (cycle detection).
        self._building: set[str] = set()
        # Names that have been fully resolved in phase 2.
        self._built: set[str] = set()

    def collect(self, program: Program) -> None:
        """Scan *program* and populate ``self._env``."""
        # ----------------------------------------------------------------
        # Phase 1: Register names and empty shells (order-independent).
        # ----------------------------------------------------------------
        for stmt in program.body:
            if isinstance(stmt, RecordDef):
                self._register_name(stmt.name, stmt.span)
                # Register an empty shell so forward references resolve.
                self._env.register_type(stmt.name, RecordType(name=stmt.name, fields={}))
                self._record_defs[stmt.name] = stmt
            elif isinstance(stmt, EnumDef):
                self._register_name(stmt.name, stmt.span)
                # Register an empty shell so forward references resolve.
                self._env.register_type(stmt.name, EnumType(name=stmt.name, variants={}))
                self._enum_defs[stmt.name] = stmt
            elif isinstance(stmt, TypeAlias):
                self._register_name(stmt.name, stmt.span)
                self._env.register_alias(stmt.name, stmt.type_expr)

        # ----------------------------------------------------------------
        # Phase 2: Resolve all field/variant types with recursion detection.
        # ----------------------------------------------------------------
        for stmt in program.body:
            if isinstance(stmt, RecordDef):
                self._ensure_built_record(stmt.name)
            elif isinstance(stmt, EnumDef):
                self._ensure_built_enum(stmt.name)
            elif isinstance(stmt, TypeAlias):
                # Aliases are resolved lazily via TypeEnvironment.resolve_type_expr;
                # validate them now to surface cycle/unknown errors early.
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

    def _ensure_built_record(self, name: str) -> None:
        """Build the record type for *name* if not already built."""
        if name in self._built:
            return
        stmt = self._record_defs[name]
        if name in self._building:
            raise AglTypeError(
                f"Record type '{name}' is directly or indirectly recursive. "
                "Recursive types are not supported in v1.",
                span=self._declared[name],
            )
        self._building.add(name)
        self._build_record(stmt)
        self._building.discard(name)
        self._built.add(name)

    def _ensure_built_enum(self, name: str) -> None:
        """Build the enum type for *name* if not already built."""
        if name in self._built:
            return
        stmt = self._enum_defs[name]
        if name in self._building:
            raise AglTypeError(
                f"Enum type '{name}' is directly or indirectly recursive. "
                "Recursive types are not supported in v1.",
                span=self._declared[name],
            )
        self._building.add(name)
        self._build_enum(stmt)
        self._building.discard(name)
        self._built.add(name)

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
        # Replace the empty shell with the fully-resolved record type.
        self._env.register_type(stmt.name, RecordType(name=stmt.name, fields=fields))

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
        # Replace the empty shell with the fully-resolved enum type.
        self._env.register_type(stmt.name, EnumType(name=stmt.name, variants=variants))

    def _resolve_field_type(self, fd: FieldDef, owner: str) -> Type:
        """Resolve a field's TypeExpr to a semantic Type.

        Before resolving, ensure that any user-declared named type referenced
        by this field has itself been fully built (triggering on-demand
        topological ordering and cycle detection).
        """
        # Determine the named type(s) referenced, and ensure they are built.
        # We only need to recurse into list/dict wrappers to find the NameT.
        self._ensure_referenced_type_built(fd.type_expr)
        return self._env.resolve_type_expr(fd.type_expr, span=fd.span)

    def _ensure_referenced_type_built(
        self, type_expr: object, _alias_seen: frozenset[str] = frozenset()
    ) -> None:
        """Recursively ensure that all user-declared types in *type_expr* are built.

        A ``NameT`` may name a record, an enum, or a ``type`` alias.  Aliases are
        transparent, so recursion that hops through one — e.g. ``type T = R`` with
        ``record R { t: T }`` — is still recursion into ``R`` and must trip the
        ``_building`` guard.  When a ``NameT`` names an alias we therefore recurse
        into the alias's raw target ``TypeExpr``.  ``_alias_seen`` guards against
        pure alias-alias cycles (``type A = B``/``type B = A``), which are diagnosed
        separately by ``_validate_alias``; here we simply stop following them so the
        on-demand build cannot loop forever.
        """
        from agm.agl.syntax.types import DictT, ListT, NameT

        if isinstance(type_expr, NameT):
            name = type_expr.name
            if name in self._record_defs:
                self._ensure_built_record(name)
            elif name in self._enum_defs:
                self._ensure_built_enum(name)
            elif self._env.get_alias_target_expr(name) is not None:
                # Transparent alias: follow its raw target so recursion routed
                # through the alias still reaches the underlying record/enum and
                # trips the ``_building`` cycle guard.
                if name not in _alias_seen:
                    self._ensure_referenced_type_built(
                        self._env.get_alias_target_expr(name),
                        _alias_seen | {name},
                    )
            # Built-in names (text, int, etc.) and unknown names are handled by
            # resolve_type_expr; we only handle user declarations here.
        elif isinstance(type_expr, ListT):
            self._ensure_referenced_type_built(type_expr.elem, _alias_seen)
        elif isinstance(type_expr, DictT):
            self._ensure_referenced_type_built(type_expr.value, _alias_seen)
        # Primitive types (TextT, IntT, etc.) have no nested declarations.

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles."""
        self._env.resolve_type_expr(stmt.type_expr, span=stmt.span)


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
        self._call_sites: list[CallSiteRecord] = []
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
        self._require_bool_condition(cond_type, stmt.condition.span, "until")

    def _check_if(self, stmt: IfStmt) -> None:
        for branch in stmt.branches:
            if not isinstance(branch.cond, ElseSentinel):
                cond_type = self._check_expr(branch.cond, expected=None)
                self._require_bool_condition(cond_type, branch.cond.span, "if")
            for s in branch.body:
                self._check_stmt(s)

    def _require_bool_condition(self, cond_type: Type, span: SourceSpan, kw: str) -> None:
        """Reject a non-bool ``if``/``until`` condition (design §4.3).

        The span points at the condition expression so the diagnostic lands on
        the offending operand rather than the enclosing statement.
        """
        if not isinstance(cond_type, BoolType):
            raise AglTypeError(
                f"'{kw}' condition must be bool; got '{cond_type!r}'.",
                span=span,
            )

    def _check_case_stmt(self, stmt: CaseStmt) -> None:
        subj_type = self._check_expr(stmt.subject, expected=None)
        for branch in stmt.branches:
            self._check_case_stmt_branch(branch, subj_type)
        self._warn_non_exhaustive(
            subj_type, [b.pattern for b in stmt.branches], stmt.span
        )

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
        # The operand must be an exception value (design §8.3). Constructing the
        # abstract ``Exception`` base is rejected at the constructor level
        # (``_check_unqualified_constructor``); a *rethrow* of an
        # ``Exception``-typed binder (e.g. ``catch _ as e => raise e``) is legal
        # and must NOT be rejected here.
        exc_type = self._check_expr(stmt.exc, expected=None)
        if not isinstance(exc_type, ExceptionType):
            raise AglTypeError(
                f"'raise' requires an exception value; got '{exc_type!r}'.",
                span=stmt.exc.span,
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
            operand_type = self._check_expr(expr.operand, expected=None)
            if not isinstance(operand_type, BoolType):
                raise AglTypeError(
                    f"'not' requires a bool operand; got '{operand_type!r}'.",
                    span=expr.operand.span,
                )
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
                # Pass ``json`` as the expected type for interpolation segments:
                # a dict/list literal inside ``${ … }`` may have mixed-kind
                # values (e.g. ``{kind: "demo", tags: items}`` where one value
                # is text and another is list[text]).  With ``expected=json``
                # the dict-literal check accepts any JSON-shaped element type
                # (design §5.8 rule 3).  A renderer override narrows the type
                # expectation further below.
                seg_type = self._check_expr(seg.expr, expected=JsonType())
                # Validate renderer name if explicit.
                if seg.render is not None and seg.render != "default":
                    if seg.render not in self._caps.renderer_names:
                        raise AglTypeError(
                            f"Unknown renderer '{seg.render}'. "
                            f"Known renderers: {sorted(self._caps.renderer_names)}.",
                            span=seg.span,
                        )
                    # A registered renderer may restrict the type kinds it
                    # accepts (F6, plan §9.1).  ``None`` means type-agnostic
                    # (accepts every kind — built-ins and renderers registered
                    # without ``supported_types``).
                    supported = self._caps.renderer_kinds.get(seg.render)
                    if supported is not None and seg_type.kind not in supported:
                        raise AglTypeError(
                            f"Renderer '{seg.render}' does not support value of "
                            f"type '{seg_type!r}' (kind '{seg_type.kind}'). "
                            f"Supported kinds: {sorted(supported)}.",
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

        # ``exec`` (shell) calls are rejected statically unless the host supports
        # them.  In M1 ``supports_shell_exec`` is always False (M4 flips it), so
        # any ``exec`` call is a static error at the call's span.
        if kind == CallKind.shell_exec and not self._caps.supports_shell_exec:
            raise AglTypeError(
                "The host does not support 'exec' calls (shell execution lands "
                "in M4).",
                span=node.span,
            )

        # Determine target type from context, defaulting to text (design §2.4).
        if expected is not None:
            target_type: Type = expected
        else:
            target_type = TextType()

        # Select codec: auto-select from capabilities, or honour an explicit
        # ``format`` option.  Codec selection and option validation are structural
        # properties of the call (independent of agent availability), so they are
        # checked before the agent-capability checks below.
        if node.options.format is not None:
            codec_name = self._validate_format_option(
                node.options.format, target_type, node.options.span
            )
        else:
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

        # Record the static call-site descriptor for the §10.1 dry-run inventory.
        # Everything the inventory needs about the call form (callee, parse policy,
        # span) is in hand here; recording it now avoids a second AST walk at
        # dry-run time.  Appending in check order preserves source order.
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee=node.agent,
                parse_policy=_parse_policy_str(node.options.parse_policy),
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )

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

    def _validate_format_option(
        self, format_name: str, target_type: Type, span: SourceSpan
    ) -> str:
        """Validate an explicit ``format`` call option and return the codec name.

        Checks:
        1. The named codec is registered in capabilities.
        2. The codec supports the call's target type.

        Raises ``AglTypeError`` on either violation.
        """
        if format_name not in self._caps.codec_kinds:
            known = sorted(self._caps.codec_kinds)
            raise AglTypeError(
                f"Unknown codec '{format_name}' in 'format' option. "
                f"Known codecs: {known}.",
                span=span,
            )
        supported_kinds = self._caps.codec_kinds[format_name]
        if target_type.kind not in supported_kinds:
            raise AglTypeError(
                f"Codec '{format_name}' does not support target type '{target_type!r}'. "
                f"(Supported kinds: {sorted(supported_kinds)}.)",
                span=span,
            )
        return format_name

    # --- Binary operations ---

    def _check_binary_op(self, node: BinaryOp) -> Type:
        left_type = self._check_expr(node.left, expected=None)
        right_type = self._check_expr(node.right, expected=None)

        op = node.op

        if op in (BinOp.AND, BinOp.OR):
            # 'and'/'or' require bool operands (design §4.3). Report the span of
            # the first offending operand.
            op_name = "and" if op is BinOp.AND else "or"
            if not isinstance(left_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; left operand has type "
                    f"'{left_type!r}'.",
                    span=node.left.span,
                )
            if not isinstance(right_type, BoolType):
                raise AglTypeError(
                    f"'{op_name}' requires bool operands; right operand has type "
                    f"'{right_type!r}'.",
                    span=node.right.span,
                )
            return BoolType()

        if op in (BinOp.EQ, BinOp.NEQ):
            # Operands must be the same type after int→decimal widening
            # (design §5.8 rule 4).  ``json`` compares only with ``json``;
            # ``json`` vs a non-``json`` type is a static error (``comparable_types``
            # does not absorb JSON-shaped scalars the way ``is_assignable`` does).
            if not comparable_types(left_type, right_type):
                raise AglTypeError(
                    f"Equality operands must have the same type; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE):
            # Ordering: numeric (int/decimal) or text; bool not allowed
            # (design §4.3 — "compare two numbers or two text values lexicographically").
            numeric_pair = isinstance(left_type, (IntType, DecimalType)) and isinstance(
                right_type, (IntType, DecimalType)
            )
            text_pair = isinstance(left_type, TextType) and isinstance(right_type, TextType)
            if not (numeric_pair or text_pair):
                raise AglTypeError(
                    f"Ordering operators require both operands to be numeric "
                    f"(int or decimal) or both to be text; "
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
        # A qualifier (``status is Status.Pass``) is resolved alias-transparently
        # and must name the same enum as the operand (design §5.4).
        if node.qualifier is not None:
            self._check_variant_qualifier(node.qualifier, expr_type, node.span)
        # Check variant membership.
        if node.variant not in expr_type.variants:
            raise AglTypeError(
                f"Variant '{node.variant}' does not belong to enum '{expr_type.name}'.",
                span=node.span,
            )
        return BoolType()

    def _check_variant_qualifier(
        self, qualifier: str, enum_type: EnumType, span: SourceSpan
    ) -> None:
        """Validate an alias-transparent enum qualifier (design §5.4).

        ``qualifier`` must resolve (through any alias chain) to ``enum_type``;
        a non-enum or mismatched qualifier is a static error.
        """
        resolved = self._env.resolve_named_type(qualifier)
        if not isinstance(resolved, EnumType):
            raise AglTypeError(
                f"'{qualifier}' is not a known enum type.",
                span=span,
            )
        if resolved.name != enum_type.name:
            raise AglTypeError(
                f"Qualifier '{qualifier}' resolves to enum '{resolved.name}', "
                f"but the value has enum type '{enum_type.name}'.",
                span=span,
            )

    # --- case expression ---

    def _check_case_expr(self, node: CaseExpr, *, expected: Type | None) -> Type:
        subj_type = self._check_expr(node.subject, expected=None)
        branch_types: list[Type] = []
        for branch in node.branches:
            self._bind_pattern_types(branch.pattern, subj_type, branch)
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)
        self._warn_non_exhaustive(
            subj_type, [b.pattern for b in node.branches], node.span
        )

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
            # Qualified: Enum.Variant — the qualifier is resolved
            # alias-transparently (design §5.4), so ``Status.Pass`` works when
            # ``type Status = Review``.  The resolved ``EnumType`` is recorded as
            # this node's type (via the return value), so the interpreter reads
            # the resolved enum rather than re-resolving the alias.
            enum_type = self._env.resolve_named_type(node.qualifier)
            if not isinstance(enum_type, EnumType):
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
        # Try exception types (built-in constructors such as Abort, MatchError).
        exc = self._env.get_type(name)
        if isinstance(exc, ExceptionType):
            # The abstract "Exception" base is not constructible.
            if exc.name == "Exception":
                raise AglTypeError(
                    "The abstract 'Exception' base type is not constructible. "
                    "Use a concrete exception type (e.g. 'Abort').",
                    span=node.span,
                )
            return self._check_exception_call(node, exc)
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

    def _check_exception_call(self, node: Constructor, exc: ExceptionType) -> ExceptionType:
        """Type-check a concrete exception constructor call (e.g. ``Abort(message: "…")``)."""
        provided = {arg.name: arg for arg in node.args}
        # Check for unknown fields.
        # Note: duplicate argument names are rejected earlier by the parser.
        for arg_name in provided:
            if arg_name not in exc.fields:
                raise AglTypeError(
                    f"Exception type '{exc.name}' has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )
        # Check for missing required fields.  The ``message`` field is required
        # for all exception types; ``trace_id`` is injected by the runtime, so
        # it may be omitted from the constructor call.
        for field_name, field_type in exc.fields.items():
            if field_name == "trace_id":
                continue  # injected by the runtime; not required in source
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call for "
                    f"exception type '{exc.name}'.",
                    span=node.span,
                )
        # Type-check supplied arguments.  By this point, every arg.name in
        # node.args is a known field (unknown fields raised above).
        for arg in node.args:
            expected_field_type = exc.fields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)
        return exc

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

    def _expected_elem_type(self, expected: Type | None) -> Type | None:
        """The element type a list literal must satisfy under *expected*.

        ``list[T]`` propagates ``T``; ``json`` propagates ``json`` (a JSON list's
        elements are themselves ``json``).  Any other context gives no
        expectation (the literal infers its own element type).
        """
        if isinstance(expected, ListType):
            return expected.elem
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _expected_value_type(self, expected: Type | None) -> Type | None:
        """The value type a dict literal must satisfy under *expected*.

        ``dict[text, V]`` propagates ``V``; ``json`` propagates ``json``.
        """
        if isinstance(expected, DictType):
            return expected.value
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _check_list_lit(self, node: ListLit, *, expected: Type | None) -> Type:
        elem_expected = self._expected_elem_type(expected)
        # Empty list: needs an expected element type to determine its type.
        if not node.elements:
            if elem_expected is None:
                raise AglTypeError(
                    "Empty list literal requires a type annotation "
                    "(e.g. 'let xs: list[text] = []').",
                    span=node.span,
                )
            return ListType(elem=elem_expected)
        if elem_expected is not None:
            # Expected-type propagation: every element must satisfy the target
            # element type (design §5.7/§5.8).
            for elem in node.elements:
                et = self._check_expr(elem, expected=elem_expected)
                self._assert_assignable(et, elem_expected, elem.span)
            return ListType(elem=elem_expected)
        # No expectation: unify all elements (int → decimal widening) and assert
        # every element is assignable to the unified type (soundness).
        unified = self._unify_elements(
            node.elements,
            kind="List",
            span=node.span,
        )
        return ListType(elem=unified)

    def _check_dict_lit(self, node: DictLit, *, expected: Type | None) -> Type:
        seen_keys: dict[str, SourceSpan] = {}
        for entry in node.entries:
            key = entry.key.value
            if key in seen_keys:
                raise AglTypeError(
                    f"Duplicate key '{key}' in dict literal.",
                    span=entry.span,
                )
            seen_keys[key] = entry.span
        val_expected = self._expected_value_type(expected)
        if not node.entries:
            if val_expected is None:
                raise AglTypeError(
                    "Empty dict literal requires a type annotation.",
                    span=node.span,
                )
            return DictType(value=val_expected)
        if val_expected is not None:
            for entry in node.entries:
                et = self._check_expr(entry.value, expected=val_expected)
                self._assert_assignable(et, val_expected, entry.span)
            return DictType(value=val_expected)
        unified = self._unify_elements(
            [entry.value for entry in node.entries],
            kind="Dict",
            span=node.span,
        )
        return DictType(value=unified)

    def _unify_elements(
        self, elements: Sequence[Expr], *, kind: str, span: SourceSpan
    ) -> Type:
        """Unify literal element types with int → decimal widening.

        Soundness: returns the common type and asserts every element is
        assignable to it.  ``[1, 2.5]`` unifies to ``decimal``; mixed
        incompatible element types are a static error.
        """
        types = [self._check_expr(e, expected=None) for e in elements]
        unified = types[0]
        for t in types[1:]:
            if is_assignable(unified, t):
                # ``t`` is the wider type (e.g. int then decimal → decimal).
                unified = t
            elif not is_assignable(t, unified):
                raise AglTypeError(
                    f"{kind} literal elements have inconsistent types: "
                    f"'{unified!r}' and '{t!r}'.",
                    span=span,
                )
        return unified

    # ------------------------------------------------------------------
    # Pattern binding helpers
    # ------------------------------------------------------------------

    def _bind_pattern_types(self, pattern: object, subj_type: Type, owner: object) -> None:
        """Record binding types for variables introduced by *pattern*."""
        if isinstance(pattern, WildcardPattern):
            pass
        elif isinstance(pattern, LiteralPattern):
            # A literal pattern's type must be compatible with the scrutinee's
            # static type (design §6.1/§11.9, F3/F5 ruling): same type after
            # int→decimal widening.  An int literal against a text scrutinee, or
            # any scalar literal against a json scrutinee, can never match and is
            # a static error (consistent with equality rule 4).
            lit_type = self._check_expr(pattern.literal, expected=None)
            if not comparable_types(lit_type, subj_type):
                raise AglTypeError(
                    f"Literal pattern of type '{lit_type!r}' is incompatible with "
                    f"scrutinee of type '{subj_type!r}'.",
                    span=pattern.span,
                )
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
            # An alias-qualified pattern (``Status.Pass``) is resolved
            # alias-transparently and must name the operand's enum (design §5.4).
            if pattern.qualifier is not None:
                self._check_variant_qualifier(pattern.qualifier, subj_type, pattern.span)
            # The variant name must belong to the scrutinee's enum (design §6.1,
            # F4) — mirroring ``_check_is_test``.  A phantom variant is a static
            # error, not a silently-dead arm that miscounts toward exhaustiveness.
            variant_name = pattern.name
            if variant_name not in subj_type.variants:
                raise AglTypeError(
                    f"Variant '{variant_name}' does not belong to enum "
                    f"'{subj_type.name}'.",
                    span=pattern.span,
                )
            # Look up variant field types and bind each sub-pattern.
            vfields = subj_type.variants[variant_name]
            # Check for duplicate pattern fields (design §5.9).
            seen_pf: set[str] = set()
            for pf in pattern.fields:
                if pf.name in seen_pf:
                    raise AglTypeError(
                        f"Duplicate field '{pf.name}' in pattern for variant "
                        f"'{variant_name}' — each field may appear at most once.",
                        span=pf.span,
                    )
                seen_pf.add(pf.name)
                if pf.name not in vfields:
                    raise AglTypeError(
                        f"Variant '{variant_name}' has no field '{pf.name}'.",
                        span=pf.span,
                    )
                field_type = vfields[pf.name]
                self._bind_pattern_types(pf.pattern, field_type, pf)

    def _warn_non_exhaustive(
        self, subj_type: Type, patterns: list[Pattern], span: SourceSpan
    ) -> None:
        """Emit a warning when an enum ``case`` leaves some variants uncovered.

        Exhaustiveness is a *warning*, not an error (plan Q4): the program still
        runs (an unmatched value raises ``MatchError`` at runtime).  Only enum
        scrutinees are analysed — for any other type the set of inhabitants is
        not enumerable, so no warning is produced.  A wildcard ``_`` or a bare
        variable pattern covers the entire remainder, so its presence suppresses
        the warning.  Otherwise the uncovered variants are the enum variants not
        named by any constructor pattern.
        """
        if not isinstance(subj_type, EnumType):
            return
        covered: set[str] = set()
        for pattern in patterns:
            if isinstance(pattern, (WildcardPattern, VarPattern)):
                # A catch-all binding covers every remaining variant.
                return
            # Only constructor patterns reach exhaustiveness analysis on an enum:
            # ``_bind_pattern_types`` rejects a literal pattern against an enum
            # scrutinee before this runs (F4/F5), so ``pattern`` is always a
            # ``ConstructorPattern`` here.
            assert isinstance(pattern, ConstructorPattern), (
                f"unexpected pattern kind on enum scrutinee: {type(pattern).__name__}"
            )
            covered.add(pattern.name)
        missing = [name for name in subj_type.variants if name not in covered]
        if not missing:
            return
        self._warnings.append(
            Diagnostic(
                message=(
                    f"Non-exhaustive case on enum '{subj_type.name}': missing "
                    f"variant(s) {', '.join(missing)}. An unmatched value raises "
                    f"MatchError at runtime."
                ),
                line=span.start_line,
                severity="warning",
            )
        )

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
            call_sites=tuple(self._call_sites),
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
