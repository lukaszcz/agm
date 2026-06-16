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
8.  Agent names are NOT validated here: the scope pass owns name validity (an
    undeclared named agent is a scope binding error).  The built-in ``ask``
    call still requires ``has_default_agent`` to back it.
9.  Assignability (design §5.8): ``int`` widens to ``decimal``; ``json``
    accepts any JSON-shaped value (scalars and ``list``/``dict`` thereof, but
    not records/enums/exceptions).  List/dict literals propagate the expected
    element/value type and assert every element soundly.  Qualified enum
    constructors, ``is`` tests, and patterns resolve their qualifier
    alias-transparently (§5.4).
10. Duplicate constructor argument names: the parser rejects these; the
    checker repeats the check defensively for direct AST construction.
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
    ConfigPragma,
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
    IfExpr,
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
        #
        # Duplicate-name rejection keys off THIS per-entry table only, which is
        # exactly the behaviour an incremental session needs: a type name that
        # exists solely in a seeded ``env`` (copied via ``TypeEnvironment.seed_from``)
        # is absent from ``self._declared``, so re-declaring it in a new entry
        # simply overwrites the seeded shell — a shadow, not a duplicate error.
        # A name declared twice WITHIN one entry still appears in ``self._declared``
        # on the second pass and is rejected; built-in names remain non-shadowable
        # via ``_BUILTIN_TYPE_NAMES``.
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
                # A legal redeclaration of a SEEDED name may change its kind;
                # drop any stale seeded entry (e.g. an alias of the same name)
                # so ``_types`` and ``_alias_targets`` stay mutually exclusive.
                self._env.unregister_name(stmt.name)
                # Register an empty shell so forward references resolve.
                self._env.register_type(stmt.name, RecordType(name=stmt.name, fields={}))
                self._record_defs[stmt.name] = stmt
            elif isinstance(stmt, EnumDef):
                self._register_name(stmt.name, stmt.span)
                self._env.unregister_name(stmt.name)
                # Register an empty shell so forward references resolve.
                self._env.register_type(stmt.name, EnumType(name=stmt.name, variants={}))
                self._enum_defs[stmt.name] = stmt
            elif isinstance(stmt, TypeAlias):
                self._register_name(stmt.name, stmt.span)
                self._env.unregister_name(stmt.name)
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
        if isinstance(stmt, (LetDecl, VarDecl)):
            self._check_binding(stmt)
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
        elif isinstance(stmt, ConfigPragma):
            pass  # Header pragma — no type-checking action needed.
        else:
            pass  # PassStmt — no-op

    def _check_binding(self, stmt: LetDecl | VarDecl) -> None:
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
        if isinstance(expr, IfExpr):
            return self._check_if_expr(expr, expected=expected)
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
                # Use the template-literal checking path for non-empty inline
                # container literals so that mixed-value-kind structures (e.g.
                # ``{kind: "demo", tags: xs}``) are accepted under design §5.8
                # rule 3, but the fabricated json expectation is confined to
                # literal STRUCTURE only and does NOT propagate into non-literal
                # child expressions (AgentCall, VarRef, CaseExpr, etc.).
                # Empty ``[]`` / ``{}`` and all other expression kinds receive
                # ``expected=None`` so that:
                # - empty literals emit the standard "needs annotation" diagnostic
                #   (design §5.6);
                # - agent calls default to the §11.4 text contract.
                is_nonempty_container_literal = (
                    isinstance(seg.expr, DictLit) and bool(seg.expr.entries)
                ) or (
                    isinstance(seg.expr, ListLit) and bool(seg.expr.elements)
                )
                if is_nonempty_container_literal:
                    assert isinstance(seg.expr, (ListLit, DictLit))
                    seg_type = self._check_template_literal(seg.expr)
                    # _check_expr is not called for the container itself in this
                    # path; record the type manually so the side-table is complete.
                    self._node_types[seg.expr.node_id] = seg_type
                else:
                    self._check_expr(seg.expr, expected=None)
        return TextType()

    def _check_template_literal(self, expr: ListLit | DictLit) -> Type:
        """Check a non-empty container literal in a ``${ … }`` context.

        Propagates ``JsonType`` only to child nodes that are themselves literals
        (scalars or container literals recursively).  Non-literal children
        (``VarRef``, ``AgentCall``, ``FieldAccess``, ``CaseExpr``, ``BinaryOp``,
        ``Constructor``, etc.) are checked with ``expected=None`` so that, in
        particular, agent calls default to the §11.4 text contract instead of
        acquiring a fabricated json contract.

        After checking each non-literal child, its inferred type is validated
        for json-assignability using the same ``is_assignable`` machinery as the
        normal literal-against-json path.
        """
        if isinstance(expr, ListLit):
            # Caller guarantees non-empty.
            for elem in expr.elements:
                self._check_template_literal_child(elem)
            return ListType(elem=JsonType())
        # DictLit — caller guarantees non-empty.
        seen_keys: dict[str, SourceSpan] = {}
        for entry in expr.entries:
            key = entry.key.value
            if key in seen_keys:
                raise AglTypeError(
                    f"Duplicate key '{key}' in dict literal.",
                    span=entry.span,
                )
            seen_keys[key] = entry.span
        for entry in expr.entries:
            self._check_template_literal_child(entry.value)
        return DictType(value=JsonType())

    def _check_template_literal_child(self, expr: Expr) -> Type:
        """Check a single child expression of a template container literal.

        Scalar literals and empty container literals receive ``expected=JsonType()``
        (structurally sound and always json-shaped).  Non-empty container literals
        are handled recursively by ``_check_template_literal`` to avoid propagating
        the fabricated json expectation into their non-literal children.
        Non-literal expressions are checked with ``expected=None`` and then their
        inferred type is asserted to be json-assignable.
        """
        if isinstance(expr, (StringLit, IntLit, DecimalLit, BoolLit, NullLit)):
            # Scalar literals: always json-shaped; check normally (expected is
            # ignored by the literal inference rules but kept for consistency).
            return self._check_expr(expr, expected=JsonType())
        if isinstance(expr, ListLit):
            if not expr.elements:
                # Empty list under json context: legal (→ list[json]).
                return self._check_expr(expr, expected=JsonType())
            # Non-empty: recurse via the template-literal path.
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        if isinstance(expr, DictLit):
            if not expr.entries:
                # Empty dict under json context: legal (→ dict[text, json]).
                return self._check_expr(expr, expected=JsonType())
            # Non-empty: recurse via the template-literal path.
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        # Non-literal: check without expectation, then assert json-assignable.
        child_type = self._check_expr(expr, expected=None)
        self._assert_assignable(child_type, JsonType(), expr.span)
        return child_type

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

        # ``exec`` (shell) calls are rejected statically unless the host declares
        # support.  ``WorkflowRuntime`` sets ``supports_shell_exec=True``; test
        # harnesses that want to forbid shell execution may set it to ``False``.
        if kind == CallKind.shell_exec and not self._caps.supports_shell_exec:
            raise AglTypeError(
                "The host does not support 'exec' (shell) calls.",
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

        # Named-agent name validity is owned by the scope pass (an undeclared
        # named agent is a binding error there); the checker does not re-validate
        # it.  Only the built-in ``ask`` call needs a backing here.
        if kind == CallKind.default_agent:
            # An ``ask`` call needs a default agent to back it.
            if not self._caps.has_default_agent:
                raise AglTypeError(
                    "No default agent is configured; the built-in 'ask' call "
                    "cannot run. Register a default agent, or run via `agm exec`, "
                    "which provides one.",
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

        # Warn about parse policies that can never take effect (design §7.2/§7.10).
        self._warn_noop_parse_policy(node, kind, target_type)

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

    def _warn_noop_parse_policy(
        self, node: AgentCall, _kind: CallKind, target_type: Type
    ) -> None:
        """Warn when an ``on_parse_error`` policy can never fire (design §7.2/§7.10).

        A policy on a *text* target is a no-op because text never fails parsing.
        Typed agent and exec calls can both fail parsing and honor their policy.
        """
        if node.options.parse_policy is None:
            return
        if isinstance(target_type, TextType):
            self._warnings.append(
                Diagnostic(
                    message=(
                        "'on_parse_error' has no effect on a text target: a text "
                        "result never fails parsing, so the policy can never fire."
                    ),
                    line=node.span.start_line,
                    severity="warning",
                )
            )

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

    # --- shared branch-type unification ---

    def _unify_branch_types(
        self,
        branch_types: list[Type],
        span: SourceSpan,
        construct: str,
    ) -> Type:
        """Unify a non-empty list of branch-body types with int→decimal widening only.

        ``construct`` is the human-readable name used in the error message
        (e.g. ``"Case expression"`` or ``"If expression"``).  All types in
        ``branch_types`` must be compatible; any pair that is not ``int``/``decimal``
        (in either order) is rejected with an ``AglTypeError``.

        Precondition: ``branch_types`` is non-empty (callers must guard on that).
        """
        result_type = branch_types[0]
        for bt in branch_types[1:]:
            if bt == result_type:
                continue
            # int + decimal (in either order) → widen to decimal.
            if isinstance(result_type, IntType) and isinstance(bt, DecimalType):
                result_type = DecimalType()
            elif isinstance(result_type, DecimalType) and isinstance(bt, IntType):
                pass  # already decimal; no change needed
            else:
                raise AglTypeError(
                    f"{construct} branches have incompatible types: "
                    f"'{result_type!r}' and '{bt!r}'.",
                    span=span,
                )
        return result_type

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

        return self._unify_branch_types(branch_types, node.span, "Case expression")

    # --- if expression ---

    def _check_if_expr(self, node: IfExpr, *, expected: Type | None) -> Type:
        """Type-check an ``IfExpr`` (if … => expr | else => expr).

        Rules (design decisions §1 and §4):
        - Each non-``else`` condition must be ``bool``.
        - An ``else`` branch is required (makes the expression total).
        - All branch-body types must unify via int→decimal widening only.
        """
        has_else = any(isinstance(b.cond, ElseSentinel) for b in node.branches)
        if not has_else:
            raise AglTypeError(
                "an `if` used as an expression must have an `else` branch",
                span=node.span,
            )
        branch_types: list[Type] = []
        for branch in node.branches:
            if not isinstance(branch.cond, ElseSentinel):
                cond_type = self._check_expr(branch.cond, expected=None)
                self._require_bool_condition(cond_type, branch.cond.span, "if")
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)
        # branch_types is always non-empty (if_expr has ≥1 branch and else is present).
        return self._unify_branch_types(branch_types, node.span, "If expression")

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
            return self._check_constructor_call(node, enum_type, variant=node.name)
        else:
            # Unqualified: look up in all known types.
            return self._check_unqualified_constructor(node, expected=expected)

    def _check_unqualified_constructor(
        self, node: Constructor, *, expected: Type | None
    ) -> Type:
        name = node.name
        # A single get_type lookup covers both record and exception types since
        # each name maps to exactly one entry in the type namespace.
        named_type = self._env.get_type(name)
        if isinstance(named_type, RecordType):
            return self._check_constructor_call(node, named_type)
        if isinstance(named_type, ExceptionType):
            # The abstract "Exception" base is not constructible.
            if named_type.abstract:
                raise AglTypeError(
                    "The abstract 'Exception' base type is not constructible. "
                    "Use a concrete exception type (e.g. 'Abort').",
                    span=node.span,
                )
            return self._check_constructor_call(node, named_type)
        # Try enums (find a unique matching variant).
        candidates: list[tuple[EnumType, str]] = []
        for type_name in self._env.all_declared_type_names():
            t = self._env.get_type(type_name)
            if isinstance(t, EnumType) and name in t.variants:
                candidates.append((t, name))
        # If expected type is an enum, resolve ambiguity.
        if isinstance(expected, EnumType) and (expected, name) in candidates:
            return self._check_constructor_call(node, expected, variant=name)
        if len(candidates) == 1:
            enum_t, variant = candidates[0]
            return self._check_constructor_call(node, enum_t, variant=variant)
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

    def _check_constructor_call(
        self,
        node: Constructor,
        owner: RecordType | EnumType | ExceptionType,
        *,
        variant: str | None = None,
    ) -> RecordType | EnumType | ExceptionType:
        """Type-check a constructor call for a record, enum variant, or exception.

        ``owner``  The record/enum/exception type being constructed.
        ``variant`` For enum types, the variant being constructed (required when
                    *owner* is an ``EnumType``).

        All three former helpers shared the same structure:
        1. Duplicate-argument check (record/enum always; exception: same rule
           applied for robustness — direct AST construction can bypass the parser
           which normally rejects duplicates first).
        2. Unknown-field check.
        3. Missing-field check (exceptions skip ``trace_id`` — runtime-injected).
        4. Argument type-check.

        The ``trace_id`` skip applies only to ``ExceptionType``; records and enum
        variants are checked for every declared field.
        """
        # Resolve the field map for the target.
        if isinstance(owner, EnumType):
            assert variant is not None, "variant is required for EnumType"
            fields = owner.variants[variant]
            type_label = f"Variant '{owner.name}.{variant}'"
        elif isinstance(owner, RecordType):
            fields = owner.fields
            type_label = f"Record '{owner.name}'"
        else:
            fields = owner.fields
            type_label = f"Exception type '{owner.name}'"

        provided = {arg.name: arg for arg in node.args}

        # 1. Duplicate-argument check (mirrors the parser guard; defensive for
        #    direct AST construction that bypasses the parser).
        seen_args: set[str] = set()
        for arg in node.args:
            if arg.name in seen_args:
                raise AglTypeError(
                    f"Duplicate argument '{arg.name}' in constructor call.",
                    span=arg.span,
                )
            seen_args.add(arg.name)

        # 2. Unknown-field check.
        for arg_name in provided:
            if arg_name not in fields:
                raise AglTypeError(
                    f"{type_label} has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )

        # 3. Missing-field check.
        for field_name in fields:
            if isinstance(owner, ExceptionType) and field_name == "trace_id":
                continue  # injected by the runtime; not required in source
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call for "
                    f"{type_label}.",
                    span=node.span,
                )

        # 4. Argument type-check.
        for arg in node.args:
            expected_field_type = fields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)

        return owner

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


def check(
    resolved: ResolvedProgram,
    capabilities: HostCapabilities,
    *,
    seed_env: TypeEnvironment | None = None,
) -> CheckedProgram:
    """Run the full M1 type-checking pass.

    Parameters
    ----------
    resolved:
        Output of the scope resolution pass.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    seed_env:
        When given, the working ``TypeEnvironment`` starts pre-populated with
        the seed's user-declared types (records/enums/aliases) and prior binding
        types (keyed by globally-unique ``decl_node_id``), so an incremental
        session entry can reference earlier declarations and bindings.  A new
        entry may shadow a seeded binding (fresh ``node_id`` → new entry) or
        replace a seeded type declaration (override, not duplicate-name error).
        Default ``None`` → today's behaviour byte-for-byte (``agm exec``).

        The returned ``CheckedProgram.type_env`` contains the *union* (seed plus
        this entry's new/overridden declarations and binding types), so the
        session can read it to promote the updated state.

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
    if seed_env is not None:
        env.seed_from(seed_env)
    program = resolved.program

    # Pre-pass: collect and validate all type declarations.
    builder = _TypeBuilder(env)
    builder.collect(program)

    # Main pass: type-check statements and expressions.
    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_program(program)

    return checker.result(resolved)
