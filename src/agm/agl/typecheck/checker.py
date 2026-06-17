"""Type-checking pass (Component 5) for AgL v2.

``check(resolved, capabilities)`` performs a bidirectional type pass over the
``ResolvedProgram``, using the ``HostCapabilities`` to validate codec and
renderer names, and returns a ``CheckedProgram``.

Rules implemented
-----------------
1.  Type declaration validation: duplicate names, unknown referenced types,
    recursive records/enums, alias cycles, and built-in-name shadowing.
2.  Function declarations: parameter/return types resolved; ordering enforced
    (required before defaulted); FunctionSignature registered.
3.  Binding type inference:
    - ``let/var name: T = e`` — check ``e`` against ``T``.
    - Other untyped initializers infer from the literal/expression.
    - ``param name[: T] [= default]`` — defaults to ``text`` when unannotated
      and defaultless; otherwise defaults are checked/inferred.
4.  ``set name = e`` — expected type is the binding's declared type.
5.  ``print(expr)`` — accepts any non-function/non-agent type.
6.  ``ask(prompt, ...)`` — named-agent or default-agent call with codec.
7.  ``exec(cmd, ...)`` — shell call; requires ``supports_shell_exec``.
8.  Declared-name calls — checked against the full ``FunctionSignature``.
9.  Value calls — checked against the ``FunctionType``; named args disallowed.
10. Lambdas — inferred or annotated return type.
11. Block typing — last item is the block's value; LetDecl/VarDecl at end is error.
12. ``if`` with no ``else`` yields ``unit``; with ``else`` branches must unify.
13. ``case`` — exhaustiveness warning on enum scrutinees.
14. ``do-until`` — yields ``unit``; condition must be bool.
15. ``try/catch`` — body and handler types must unify.
16. ``raise`` — yields ``BottomType`` (bottom, assignable to any target).
17. Assignability (design §5.8): ``int`` widens to ``decimal``; ``json``
    accepts any JSON-shaped value.  Bottom type is assignable to any target.
18. Duplicate constructor argument names, duplicate dict keys, and all the
    constructor checks carried over from v1.

The checker raises ``AglTypeError`` on the first error (first-error abort).
"""

from __future__ import annotations

from collections.abc import Sequence

from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.scope.symbols import (
    BUILTIN_CALL_NAMES,
    BinderKind,
    BindingRef,
    BuiltinKind,
    ResolvedProgram,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Call,
    Case,
    CatchClause,
    ConfigPragma,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictLit,
    Do,
    ElseSentinel,
    EnumDef,
    Expr,
    FieldAccess,
    FieldDef,
    FuncDef,
    If,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NullLit,
    ParamDecl,
    Pattern,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    SetStmt,
    StringLit,
    Template,
    Try,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
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
    FunctionSignature,
    OutputContractSpec,
    TypeEnvironment,
)
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_PRELUDE_TYPE_NAMES,
    AgentType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    Type,
    UnitType,
    comparable_types,
    is_assignable,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Built-in type names that the user may not shadow with a record/enum/alias.
_BUILTIN_TYPE_NAMES: frozenset[str] = (
    frozenset({"text", "json", "bool", "int", "decimal", "unit", "agent"})
    | BUILTIN_EXCEPTION_NAMES
    | BUILTIN_PRELUDE_TYPE_NAMES
)

# Built-in function names that user-defined defs may not shadow. Derived from
# the single source of truth in ``scope.symbols`` so the two never drift.
_BUILTIN_FUNC_NAMES: frozenset[str] = frozenset(BUILTIN_CALL_NAMES)


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
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_type(item.name, RecordType(name=item.name, fields={}))
                self._record_defs[item.name] = item
            elif isinstance(item, EnumDef):
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_type(item.name, EnumType(name=item.name, variants={}))
                self._enum_defs[item.name] = item
            elif isinstance(item, TypeAlias):
                self._register_name(item.name, item.span)
                self._env.unregister_name(item.name)
                self._env.register_alias(item.name, item.type_expr)

        # ----------------------------------------------------------------
        # Phase 2: Resolve all field/variant types with recursion detection.
        # ----------------------------------------------------------------
        for item in program.body.items:
            if isinstance(item, RecordDef):
                self._ensure_built_record(item.name)
            elif isinstance(item, EnumDef):
                self._ensure_built_enum(item.name)
            elif isinstance(item, TypeAlias):
                self._validate_alias(item)

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
        self._env.register_type(stmt.name, EnumType(name=stmt.name, variants=variants))

    def _resolve_field_type(self, fd: FieldDef, owner: str) -> Type:
        """Resolve a field's TypeExpr to a semantic Type."""
        self._ensure_referenced_type_built(fd.type_expr)
        return self._env.resolve_type_expr(fd.type_expr, span=fd.span)

    def _ensure_referenced_type_built(
        self, type_expr: object, _alias_seen: frozenset[str] = frozenset()
    ) -> None:
        """Recursively ensure that all user-declared types in *type_expr* are built."""
        from agm.agl.syntax.types import DictT, ListT, NameT

        if isinstance(type_expr, NameT):
            name = type_expr.name
            if name in self._record_defs:
                self._ensure_built_record(name)
            elif name in self._enum_defs:
                self._ensure_built_enum(name)
            elif self._env.get_alias_target_expr(name) is not None:
                if name not in _alias_seen:
                    self._ensure_referenced_type_built(
                        self._env.get_alias_target_expr(name),
                        _alias_seen | {name},
                    )
        elif isinstance(type_expr, ListT):
            self._ensure_referenced_type_built(type_expr.elem, _alias_seen)
        elif isinstance(type_expr, DictT):
            self._ensure_referenced_type_built(type_expr.value, _alias_seen)

    def _validate_alias(self, stmt: TypeAlias) -> None:
        """Validate that the alias target resolves without cycles."""
        self._env.resolve_type_expr(stmt.type_expr, span=stmt.span)


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------


class _Checker:
    """Stateful type-checking visitor for AgL v2.

    Walks the program's items in order, maintaining a binding-type lookup
    table (``node_id → Type``) populated for declarations and inline inference.

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
    # Pre-registration of function signatures
    # ------------------------------------------------------------------

    def _preregister_funcdef(self, node: FuncDef) -> None:
        """Resolve and register the signature of a top-level ``def``."""
        if node.name in _BUILTIN_TYPE_NAMES:
            raise AglTypeError(
                f"'{node.name}' is a built-in type name and cannot be used as a function name.",
                span=node.span,
            )
        if node.name in _BUILTIN_FUNC_NAMES:
            raise AglTypeError(
                f"'{node.name}' is a built-in function name and cannot be redefined.",
                span=node.span,
            )
        params: list[tuple[str, Type, bool]] = []
        seen_required = True  # True until first defaulted param
        for p in node.params:
            pt = self._env.resolve_type_expr(p.type_expr, span=p.span)
            has_default = p.default is not None
            if seen_required and has_default:
                # First defaulted param: switch to "defaulted" mode
                seen_required = False
            elif not seen_required and not has_default:
                raise AglTypeError(
                    f"Parameter '{p.name}' has no default but follows a defaulted parameter. "
                    "Required parameters must come before parameters with defaults.",
                    span=p.span,
                )
            params.append((p.name, pt, has_default))

        result_type = self._env.resolve_type_expr(node.return_type, span=node.span)
        sig = FunctionSignature(params=tuple(params), result=result_type)
        self._env.register_function_signature(node.name, sig)
        # Register the binding type as FunctionType (erases names/defaults).
        func_type = FunctionType(
            params=tuple(pt for _, pt, _ in params),
            result=result_type,
        )
        self._env.set_binding_type(node.node_id, func_type)

    # ------------------------------------------------------------------
    # Program-level check
    # ------------------------------------------------------------------

    def check_program(self, program: Program) -> None:
        """Type-check the entire program."""
        # Pre-pass: register all FuncDef signatures so calls can reference them
        # in declaration order (mutual forward references are not supported but
        # order-independence within the block is).
        for item in program.body.items:
            if isinstance(item, FuncDef):
                self._preregister_funcdef(item)

        self._check_block(program.body, expected=None)

    # ------------------------------------------------------------------
    # Block and item dispatch
    # ------------------------------------------------------------------

    def _check_block(self, block: Block, *, expected: Type | None) -> Type:
        """Type-check a block and return the type of its last item."""
        if not block.items:
            return UnitType()

        last = block.items[-1]
        if isinstance(last, (LetDecl, VarDecl)):
            raise AglTypeError(
                "a 'let'/'var' declaration must be followed by an expression in a block.",
                span=last.span,
            )

        result_type: Type = UnitType()
        for item in block.items:
            item_type = self._check_item(item, expected=expected if item is last else None)
            if item is last:
                result_type = item_type
        return result_type

    def _check_item(self, item: Item, *, expected: Type | None) -> Type:
        """Dispatch a single block item, returning its type contribution."""
        # --- Declarations ---
        if isinstance(item, FuncDef):
            # Signature already registered in pre-pass; check body now.
            self._check_funcdef_body(item)
            return UnitType()
        if isinstance(item, (RecordDef, EnumDef, TypeAlias, ConfigPragma)):
            return UnitType()
        if isinstance(item, AgentDecl):
            self._env.set_binding_type(item.node_id, AgentType())
            return UnitType()
        if isinstance(item, ParamDecl):
            self._check_param(item)
            return UnitType()
        if isinstance(item, ProgramDecl):
            return UnitType()
        # --- Binders ---
        if isinstance(item, (LetDecl, VarDecl)):
            self._check_binding(item)
            return UnitType()
        if isinstance(item, SetStmt):
            self._check_set_stmt(item)
            return UnitType()
        # --- Expr ---
        return self._check_expr(item, expected=expected)

    # ------------------------------------------------------------------
    # Declaration checkers
    # ------------------------------------------------------------------

    def _check_funcdef_body(self, node: FuncDef) -> None:
        """Check the body of a ``def`` against its registered signature."""
        sig = self._env.get_function_signature(node.name)
        assert sig is not None, f"FuncDef '{node.name}' not pre-registered"
        # Bind params in the env.
        for p, (pname, ptype, has_default) in zip(node.params, sig.params):
            self._env.set_binding_type(p.node_id, ptype)
        # Check defaults against declared parameter types.
        for p, (pname, ptype, has_default) in zip(node.params, sig.params):
            if p.default is not None:
                def_type = self._check_expr(p.default, expected=ptype)
                self._assert_assignable(def_type, ptype, p.span)
        # Check body against declared return type.
        body_type = self._check_expr(node.body, expected=sig.result)
        if not isinstance(body_type, BottomType):
            self._assert_assignable(body_type, sig.result, node.span)

    def _check_param(self, stmt: ParamDecl) -> None:
        ann_type = (
            self._env.resolve_type_expr(stmt.annotation, span=stmt.span)
            if stmt.annotation is not None
            else None
        )
        if stmt.default is not None:
            val_type = self._check_expr(stmt.default, expected=ann_type)
            if ann_type is not None:
                self._assert_assignable(val_type, ann_type, stmt.span)
                declared_type = ann_type
            else:
                if isinstance(val_type, BottomType):
                    raise AglTypeError(
                        "Cannot infer type of param: default always raises. "
                        "Add a type annotation.",
                        span=stmt.span,
                    )
                declared_type = val_type
        else:
            declared_type = ann_type if ann_type is not None else TextType()
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_binding(self, stmt: LetDecl | VarDecl) -> None:
        ann_type = self._resolve_annotation(stmt.type_ann, stmt.span)
        val_type = self._check_expr(stmt.value, expected=ann_type)
        if ann_type is not None:
            self._assert_assignable(val_type, ann_type, stmt.span)
            declared_type = ann_type
        else:
            if isinstance(val_type, BottomType):
                raise AglTypeError(
                    "Cannot infer type of binding: value always raises. Add a type annotation.",
                    span=stmt.span,
                )
            declared_type = val_type
        self._env.set_binding_type(stmt.node_id, declared_type)

    def _check_set_stmt(self, stmt: SetStmt) -> None:
        ref = self._resolved.resolution[stmt.node_id]
        target_type = self._require_binding_type(ref)
        val_type = self._check_expr(stmt.value, expected=target_type)
        self._assert_assignable(val_type, target_type, stmt.span)

    # ------------------------------------------------------------------
    # Expression type inference
    # ------------------------------------------------------------------

    def _check_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Infer/check the type of *expr*, recording it in ``_node_types``."""
        typ = self._infer_expr(expr, expected=expected)
        self._node_types[expr.node_id] = typ
        return typ

    def _infer_expr(self, expr: Expr, *, expected: Type | None) -> Type:
        """Bottom-up inference with optional top-down ``expected`` context."""
        if isinstance(expr, UnitLit):
            return UnitType()
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
        if isinstance(expr, Call):
            return self._check_call(expr, expected=expected)
        if isinstance(expr, Lambda):
            return self._check_lambda(expr, expected=expected)
        if isinstance(expr, Block):
            return self._check_block(expr, expected=expected)
        if isinstance(expr, If):
            return self._check_if(expr, expected=expected)
        if isinstance(expr, Case):
            return self._check_case(expr, expected=expected)
        if isinstance(expr, Do):
            return self._check_do(expr)
        if isinstance(expr, Try):
            return self._check_try(expr, expected=expected)
        if isinstance(expr, Raise):
            return self._infer_raise(expr)
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
        if isinstance(expr, FieldAccess):
            return self._check_field_access(expr)
        if isinstance(expr, Constructor):
            return self._check_constructor(expr, expected=expected)
        if isinstance(expr, ListLit):
            return self._check_list_lit(expr, expected=expected)
        # DictLit is the last Expr union member.
        assert isinstance(expr, DictLit), f"Unexpected expression kind: {type(expr).__name__}"
        return self._check_dict_lit(expr, expected=expected)

    # --- VarRef ---

    def _check_varref(self, node: VarRef) -> Type:
        ref = self._resolved.resolution[node.node_id]
        return self._require_binding_type(ref)

    def _require_binding_type(self, ref: BindingRef) -> Type:
        typ = self._env.resolve_binding(ref)
        if typ is None:
            raise AssertionError(
                f"Binding {ref!r} has no recorded type; checker invariant violated."
            )
        return typ

    # --- Call dispatch ---

    def _check_call(self, node: Call, *, expected: Type | None) -> Type:
        """Dispatch a Call node to the appropriate checker."""
        # Built-in?
        if node.node_id in self._resolved.builtin_calls:
            kind = self._resolved.builtin_calls[node.node_id]
            if kind == BuiltinKind.PRINT:
                return self._check_print_call(node)
            if kind == BuiltinKind.ASK:
                return self._check_ask_call(node, expected=expected)
            if kind == BuiltinKind.ASK_REQUEST:
                return self._check_ask_request_call(node)
            # EXEC
            return self._check_exec_call(node, expected=expected)

        # Declared function by name?
        # Take the declared-name (named/default) path ONLY when the callee is a
        # bare VarRef that resolves to a top-level function_binding (a ``def``).
        # A let/var-bound function value, a param, or a field access must all
        # take the value-call path — they are not declared names and do not
        # support named/defaulted arguments.
        if isinstance(node.callee, VarRef):
            callee_ref = self._resolved.resolution.get(node.callee.node_id)
            if callee_ref is not None and callee_ref.kind is BinderKind.function_binding:
                return self._check_declared_name_call(
                    node, node.callee.name, expected=expected
                )

        # Value call (lambda or higher-order).
        return self._check_value_call(node, expected=expected)

    # --- print ---

    def _check_print_call(self, node: Call) -> Type:
        if len(node.args) != 1 or node.named_args:
            raise AglTypeError(
                "print() requires exactly one positional argument.",
                span=node.span,
            )
        arg_type = self._check_expr(node.args[0], expected=None)
        if isinstance(arg_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "a function/agent value has no rendering and cannot be printed.",
                span=node.args[0].span,
            )
        return UnitType()

    # --- ask ---

    _ASK_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"agent", "format", "strict_json", "on_parse_error"}
    )

    def _check_ask_call(self, node: Call, *, expected: Type | None) -> Type:
        # Target type from context.
        target_type: Type = expected if expected is not None else TextType()

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse agent output into a function/agent value.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args.
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"ask: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Prompt (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError("ask() requires a prompt argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                "ask: too many positional arguments (expected 1).",
                span=node.span,
            )
        prompt_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(prompt_type, TextType(), node.args[0].span)

        # agent: named arg.
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._check_expr(agent_na.value, expected=None)
            if not isinstance(agent_type, AgentType):
                raise AglTypeError(
                    f"'agent:' argument must be of type agent; got '{agent_type!r}'.",
                    span=agent_na.span,
                )
        else:
            if not self._caps.has_default_agent:
                raise AglTypeError(
                    "No default agent is configured; the built-in 'ask' call "
                    "cannot run. Register a default agent, or run via `agm exec`, "
                    "which provides one.",
                    span=node.span,
                )

        codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
            node, target_type, named
        )
        spec = OutputContractSpec(
            target_type=target_type, codec_name=codec_name, strict_json=effective_strict
        )
        self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="ask",
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return target_type

    # --- ask-request ---

    def _check_ask_request_call(self, node: Call) -> Type:
        """Type-check ``ask-request(prompt, ...)`` — the side-effect-free twin of ``ask``.

        Like ``ask`` it builds an output contract from a target type and the
        parse-shaping named args (``format`` / ``strict_json`` /
        ``on_parse_error``), and accepts an ``agent:`` named arg.  But it never
        dispatches to the agent: it yields the ``AgentRequest`` record that the
        corresponding ``ask`` call would pass to ``AgentRegistry.dispatch`` on
        its first attempt.

        The target type is taken from the explicit type argument
        (``ask-request::[Review](...)``) when present, and defaults to ``text``
        otherwise (``ask-request(...)``).  Because the result type is fixed to
        ``AgentRequest``, the contextual ``expected`` type is ignored — unlike
        ``ask``, the target type is not inferred from context.
        """
        agent_request_type = self._env.get_type("AgentRequest")
        assert agent_request_type is not None, "AgentRequest prelude type missing"

        # Target type: explicit type argument, else text default.
        if node.type_arg is not None:
            target_type = self._env.resolve_type_expr(node.type_arg, span=node.span)
        else:
            target_type = TextType()

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot build an output contract for a function/agent target.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args (same set as ask, minus none — agent is
        # accepted even though it only labels the request, never dispatches).
        for arg_name, na in named.items():
            if arg_name not in self._ASK_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"ask-request: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Prompt (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError(
                "ask-request() requires a prompt argument.", span=node.span
            )
        if len(node.args) > 1:
            raise AglTypeError(
                "ask-request: too many positional arguments (expected 1).",
                span=node.span,
            )
        prompt_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(prompt_type, TextType(), node.args[0].span)

        # agent: named arg — same validation as ask (must be an agent value).
        if "agent" in named:
            agent_na = named["agent"]
            agent_type = self._check_expr(agent_na.value, expected=None)
            if not isinstance(agent_type, AgentType):
                raise AglTypeError(
                    f"'agent:' argument must be of type agent; got '{agent_type!r}'.",
                    span=agent_na.span,
                )

        # Build the same output contract spec an ``ask`` call would, so the
        # materialized contract (and thus the returned request) matches exactly.
        codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
            node, target_type, named
        )
        spec = OutputContractSpec(
            target_type=target_type, codec_name=codec_name, strict_json=effective_strict
        )
        self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="ask-request",
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return agent_request_type

    # --- exec ---

    _EXEC_ALLOWED_NAMED_ARGS: frozenset[str] = frozenset(
        {"format", "strict_json", "on_parse_error"}
    )

    def _check_exec_call(self, node: Call, *, expected: Type | None) -> Type:
        if not self._caps.supports_shell_exec:
            raise AglTypeError(
                "The host does not support 'exec' (shell) calls.", span=node.span
            )

        exec_result_type = self._env.get_type("ExecResult")
        target_type: Type
        if expected is not None:
            target_type = expected
        else:
            assert exec_result_type is not None
            target_type = exec_result_type

        # D9: reject function/agent targets.
        if isinstance(target_type, (FunctionType, AgentType)):
            raise AglTypeError(
                "cannot parse exec output into a function/agent value.",
                span=node.span,
            )

        named = {na.name: na for na in node.named_args}

        # Reject unknown named args (exec has no 'agent:' argument).
        for arg_name, na in named.items():
            if arg_name not in self._EXEC_ALLOWED_NAMED_ARGS:
                raise AglTypeError(
                    f"exec: unknown argument '{arg_name}'.",
                    span=na.span,
                )

        # Command (first positional arg — reject extra positionals).
        if not node.args:
            raise AglTypeError("exec() requires a command argument.", span=node.span)
        if len(node.args) > 1:
            raise AglTypeError(
                "exec: too many positional arguments (expected 1).",
                span=node.span,
            )
        cmd_type = self._check_expr(node.args[0], expected=TextType())
        self._assert_assignable(cmd_type, TextType(), node.args[0].span)

        # Determine codec.
        is_exec_result = exec_result_type is not None and target_type == exec_result_type
        parse_policy_str = "default"

        if is_exec_result:
            # Structured form: reject parse-shaping options — they are meaningless
            # when exec returns the raw ExecResult record.
            for shaping_arg in ("format", "strict_json", "on_parse_error"):
                if shaping_arg in named:
                    raise AglTypeError(
                        f"exec returning ExecResult does not accept '{shaping_arg}'; "
                        "those options apply only when parsing stdout into a typed value.",
                        span=named[shaping_arg].span,
                    )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name="text",
                strict_json=None,
                structured_exec=True,
            )
        else:
            codec_name, effective_strict, parse_policy_str = self._resolve_parse_options(
                node, target_type, named
            )
            spec = OutputContractSpec(
                target_type=target_type,
                codec_name=codec_name,
                strict_json=effective_strict,
            )
        self._contract_specs[node.node_id] = spec
        self._call_sites.append(
            CallSiteRecord(
                node_id=node.node_id,
                callee="exec",
                parse_policy=parse_policy_str,
                line=node.span.start_line,
                col=node.span.start_col,
            )
        )
        return target_type

    # --- shared parse-option handling (ask / exec) ---

    def _resolve_parse_options(
        self, node: Call, target_type: Type, named: dict[str, NamedArg]
    ) -> tuple[str, bool | None, str]:
        """Resolve the format/strict_json/on_parse_error named args shared by ask and exec.

        Returns ``(codec_name, effective_strict, parse_policy_str)``.
        """
        if "format" in named:
            format_na = named["format"]
            fmt_expr = format_na.value
            if not isinstance(fmt_expr, StringLit):
                raise AglTypeError(
                    "'format' must be a static text literal (codec name).",
                    span=format_na.span,
                )
            codec_name = self._validate_format_option(fmt_expr.value, target_type, format_na.span)
        else:
            codec_name = self._select_codec(target_type, node.span)

        strict_json: bool | None = None
        if "strict_json" in named:
            sj_na = named["strict_json"]
            sj_expr = sj_na.value
            if not isinstance(sj_expr, BoolLit):
                raise AglTypeError(
                    "'strict_json' must be a static bool literal.",
                    span=sj_na.span,
                )
            if codec_name != "json":
                raise AglTypeError(
                    f"'strict_json' is only valid when the codec is 'json'; "
                    f"the selected codec for this call is '{codec_name}'.",
                    span=sj_na.span,
                )
            strict_json = sj_expr.value

        parse_policy_str = "default"
        if "on_parse_error" in named:
            ope_na = named["on_parse_error"]
            parse_policy_str = self._extract_parse_policy_str(ope_na.value, ope_na.span)
            # Warn: no-op on text target.
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

        effective_strict = strict_json if codec_name == "json" else None
        return codec_name, effective_strict, parse_policy_str

    # --- on_parse_error policy extraction ---

    def _extract_parse_policy_str(self, arg: Expr, span: SourceSpan) -> str:
        """Extract a static ``ParsePolicy`` constructor as an inventory string."""
        if isinstance(arg, Constructor):
            qualifier = arg.qualifier
            if qualifier is not None and qualifier != "ParsePolicy":
                raise AglTypeError(
                    "'on_parse_error' must be a static ParsePolicy constructor "
                    "(Abort or Retry(n: <int>)).",
                    span=span,
                )
            if arg.name == "Abort":
                if arg.args:
                    raise AglTypeError(
                        "'on_parse_error' must be a static ParsePolicy constructor "
                        "(Abort or Retry(n: <int>)).",
                        span=span,
                    )
                return "abort"
            if arg.name == "Retry":
                n_arg = next((a for a in arg.args if a.name == "n"), None)
                if n_arg is None or not isinstance(n_arg.value, IntLit):
                    raise AglTypeError(
                        "'on_parse_error' must be a static ParsePolicy constructor "
                        "(Abort or Retry(n: <int>)).",
                        span=span,
                    )
                return f"retry[{n_arg.value.value}]"
        raise AglTypeError(
            "'on_parse_error' must be a static ParsePolicy constructor "
            "(Abort or Retry(n: <int>)).",
            span=span,
        )

    # --- codec helpers ---

    def _select_codec(self, target_type: Type, span: SourceSpan) -> str:
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

    # --- declared-name call ---

    def _check_declared_name_call(
        self, node: Call, func_name: str, *, expected: Type | None
    ) -> Type:
        sig = self._env.get_function_signature(func_name)
        if sig is None:
            return self._check_value_call(node, expected=expected)

        if len(node.args) > len(sig.params):
            raise AglTypeError(
                f"Too many positional arguments: '{func_name}' takes "
                f"{len(sig.params)} parameter(s), got {len(node.args)}.",
                span=node.span,
            )

        positional_filled: set[str] = set()
        for i, arg in enumerate(node.args):
            pname, ptype, _ = sig.params[i]
            at = self._check_expr(arg, expected=ptype)
            self._assert_assignable(at, ptype, arg.span)
            positional_filled.add(pname)

        named_filled: set[str] = set()
        for na in node.named_args:
            match: tuple[str, Type, bool] | None = None
            for p in sig.params:
                if p[0] == na.name:
                    match = p
                    break
            if match is None:
                raise AglTypeError(
                    f"Unknown parameter '{na.name}' in call to '{func_name}'.",
                    span=na.span,
                )
            pname, ptype, has_def = match
            if pname in positional_filled:
                raise AglTypeError(
                    f"Parameter '{pname}' supplied both positionally and by name.",
                    span=na.span,
                )
            if pname in named_filled:
                raise AglTypeError(
                    f"Duplicate named argument '{pname}' in call to '{func_name}'.",
                    span=na.span,
                )
            at = self._check_expr(na.value, expected=ptype)
            self._assert_assignable(at, ptype, na.span)
            named_filled.add(pname)

        # Check all required params are supplied.
        for pname, ptype, has_def in sig.params:
            if not has_def and pname not in positional_filled and pname not in named_filled:
                raise AglTypeError(
                    f"Missing required argument '{pname}' in call to '{func_name}'.",
                    span=node.span,
                )

        return sig.result

    # --- value call (lambda / higher-order) ---

    def _check_value_call(self, node: Call, *, expected: Type | None) -> Type:
        if node.named_args:
            raise AglTypeError(
                "Named arguments are only allowed at declared-function call sites.",
                span=node.span,
            )
        callee_type = self._check_expr(node.callee, expected=None)
        if not isinstance(callee_type, FunctionType):
            raise AglTypeError(
                f"callee is not a function; got '{callee_type!r}'.",
                span=node.callee.span,
            )
        if len(node.args) != len(callee_type.params):
            raise AglTypeError(
                f"Arity mismatch: function expects {len(callee_type.params)} argument(s), "
                f"got {len(node.args)}.",
                span=node.span,
            )
        for arg, ptype in zip(node.args, callee_type.params):
            at = self._check_expr(arg, expected=ptype)
            self._assert_assignable(at, ptype, arg.span)
        return callee_type.result

    # --- Lambda ---

    def _check_lambda(self, node: Lambda, *, expected: Type | None) -> Type:
        param_types: list[Type] = []
        for p in node.params:
            pt = self._env.resolve_type_expr(p.type_expr, span=p.span)
            param_types.append(pt)
            self._env.set_binding_type(p.node_id, pt)

        if node.return_type is not None:
            result_type = self._env.resolve_type_expr(node.return_type, span=node.span)
            body_type = self._check_expr(node.body, expected=result_type)
            if not isinstance(body_type, BottomType):
                self._assert_assignable(body_type, result_type, node.span)
        else:
            body_type = self._check_expr(node.body, expected=None)
            if isinstance(body_type, BottomType):
                raise AglTypeError(
                    "Cannot infer return type of lambda: body always raises.",
                    span=node.span,
                )
            result_type = body_type

        return FunctionType(params=tuple(param_types), result=result_type)

    # --- if ---

    def _check_if(self, node: If, *, expected: Type | None) -> Type:
        has_else = any(isinstance(b.cond, ElseSentinel) for b in node.branches)
        branch_types: list[Type] = []
        body_expected = expected if has_else else UnitType()
        for branch in node.branches:
            if not isinstance(branch.cond, ElseSentinel):
                cond_type = self._check_expr(branch.cond, expected=None)
                self._require_bool_condition(cond_type, branch.cond.span, "if")
            bt = self._check_expr(branch.body, expected=body_expected)
            if not has_else:
                self._assert_assignable(bt, UnitType(), branch.body.span)
            branch_types.append(bt)

        if not has_else:
            return UnitType()

        return self._unify_branch_types(branch_types, node.span, "If expression")

    # --- case ---

    def _check_case(self, node: Case, *, expected: Type | None) -> Type:
        subj_type = self._check_expr(node.subject, expected=None)
        branch_types: list[Type] = []
        for branch in node.branches:
            self._bind_pattern_types(branch.pattern, subj_type, branch)
            bt = self._check_expr(branch.body, expected=expected)
            branch_types.append(bt)
        self._warn_non_exhaustive(subj_type, [b.pattern for b in node.branches], node.span)

        if not branch_types:
            return expected if expected is not None else TextType()

        return self._unify_branch_types(branch_types, node.span, "Case expression")

    # --- do ---

    def _check_do(self, node: Do) -> Type:
        self._check_expr(node.body, expected=None)
        cond_type = self._check_expr(node.condition, expected=None)
        self._require_bool_condition(cond_type, node.condition.span, "do-until")
        return UnitType()

    # --- try ---

    def _check_try(self, node: Try, *, expected: Type | None) -> Type:
        body_type = self._check_expr(node.body, expected=expected)
        handler_types: list[Type] = [body_type]
        for clause in node.handlers:
            ht = self._check_catch_clause(clause, expected=expected)
            handler_types.append(ht)
        return self._unify_branch_types(handler_types, node.span, "Try expression")

    def _check_catch_clause(self, clause: CatchClause, *, expected: Type | None) -> Type:
        if clause.exc_type is None or clause.exc_type == "_":
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
        if clause.binding is not None:
            self._env.set_binding_type(clause.node_id, exc_type)
        return self._check_expr(clause.body, expected=expected)

    # --- raise ---

    def _infer_raise(self, node: Raise) -> BottomType:
        exc_type = self._check_expr(node.exc, expected=None)
        if not isinstance(exc_type, ExceptionType):
            raise AglTypeError(
                f"'raise' requires an exception value; got '{exc_type!r}'.",
                span=node.exc.span,
            )
        return BottomType()

    # --- template ---

    def _check_template(self, node: Template) -> TextType:
        for seg in node.segments:
            if isinstance(seg, InterpSegment):
                is_nonempty_container_literal = (
                    isinstance(seg.expr, DictLit) and bool(seg.expr.entries)
                ) or (isinstance(seg.expr, ListLit) and bool(seg.expr.elements))
                if is_nonempty_container_literal:
                    assert isinstance(seg.expr, (ListLit, DictLit))
                    seg_type = self._check_template_literal(seg.expr)
                    self._node_types[seg.expr.node_id] = seg_type
                else:
                    seg_type = self._check_expr(seg.expr, expected=None)
                    if isinstance(seg_type, (FunctionType, AgentType)):
                        raise AglTypeError(
                            "a function/agent value has no rendering and cannot be interpolated.",
                            span=seg.expr.span,
                        )
        return TextType()

    def _check_template_literal(self, expr: ListLit | DictLit) -> Type:
        """Check a non-empty container literal in a ``${ … }`` context."""
        if isinstance(expr, ListLit):
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
        """Check a single child of a template container literal."""
        if isinstance(expr, (StringLit, IntLit, DecimalLit, BoolLit, NullLit)):
            return self._check_expr(expr, expected=JsonType())
        if isinstance(expr, ListLit):
            if not expr.elements:
                return self._check_expr(expr, expected=JsonType())
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        if isinstance(expr, DictLit):
            if not expr.entries:
                return self._check_expr(expr, expected=JsonType())
            result = self._check_template_literal(expr)
            self._node_types[expr.node_id] = result
            return result
        child_type = self._check_expr(expr, expected=None)
        self._assert_assignable(child_type, JsonType(), expr.span)
        return child_type

    # --- binary ops ---

    def _check_binary_op(self, node: BinaryOp) -> Type:
        left_type = self._check_expr(node.left, expected=None)
        right_type = self._check_expr(node.right, expected=None)
        op = node.op

        if op in (BinOp.AND, BinOp.OR):
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
            if not comparable_types(left_type, right_type):
                raise AglTypeError(
                    f"Equality operands must have the same type; "
                    f"got '{left_type!r}' and '{right_type!r}'.",
                    span=node.span,
                )
            return BoolType()

        if op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE):
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
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return TextType()
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
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            return BoolType()
        if isinstance(right_type, ListType):
            if not is_assignable(left_type, right_type.elem):
                raise AglTypeError(
                    f"'in' element type mismatch: '{left_type!r}' in 'list[{right_type.elem!r}]'.",
                    span=span,
                )
            return BoolType()
        if isinstance(right_type, DictType) and isinstance(left_type, TextType):
            return BoolType()
        raise AglTypeError(
            f"'in' requires (text in text), (T in list[T]), or (text in dict); "
            f"got '{left_type!r}' in '{right_type!r}'.",
            span=span,
        )

    # --- unary ---

    def _check_unary_neg(self, node: UnaryNeg) -> Type:
        t = self._check_expr(node.operand, expected=None)
        if not isinstance(t, (IntType, DecimalType)):
            raise AglTypeError(
                f"Unary '-' requires a numeric operand; got '{t!r}'.",
                span=node.span,
            )
        return t

    # --- is test ---

    def _check_is_test(self, node: IsTest) -> BoolType:
        expr_type = self._check_expr(node.expr, expected=None)
        if not isinstance(expr_type, EnumType):
            raise AglTypeError(
                f"'is' / 'is not' requires an enum-typed left-hand side; "
                f"got '{expr_type!r}'.",
                span=node.span,
            )
        if node.qualifier is not None:
            self._check_variant_qualifier(node.qualifier, expr_type, node.span)
        if node.variant not in expr_type.variants:
            raise AglTypeError(
                f"Variant '{node.variant}' does not belong to enum '{expr_type.name}'.",
                span=node.span,
            )
        return BoolType()

    def _check_variant_qualifier(
        self, qualifier: str, enum_type: EnumType, span: SourceSpan
    ) -> None:
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

    # --- field access ---

    def _check_field_access(self, node: FieldAccess) -> Type:
        obj_type = self._check_expr(node.obj, expected=None)
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

    # --- constructor ---

    def _check_constructor(self, node: Constructor, *, expected: Type | None) -> Type:
        if node.qualifier is not None:
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
        return self._check_unqualified_constructor(node, expected=expected)

    def _check_unqualified_constructor(
        self, node: Constructor, *, expected: Type | None
    ) -> Type:
        name = node.name
        named_type = self._env.get_type(name)
        if isinstance(named_type, RecordType):
            return self._check_constructor_call(node, named_type)
        if isinstance(named_type, ExceptionType):
            if named_type.abstract:
                raise AglTypeError(
                    "The abstract 'Exception' base type is not constructible. "
                    "Use a concrete exception type (e.g. 'Abort').",
                    span=node.span,
                )
            return self._check_constructor_call(node, named_type)
        candidates: list[tuple[EnumType, str]] = []
        for type_name in self._env.all_declared_type_names():
            t = self._env.get_type(type_name)
            if isinstance(t, EnumType) and name in t.variants:
                candidates.append((t, name))
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

        seen_args: set[str] = set()
        for arg in node.args:
            if arg.name in seen_args:
                raise AglTypeError(
                    f"Duplicate argument '{arg.name}' in constructor call.",
                    span=arg.span,
                )
            seen_args.add(arg.name)

        for arg_name in provided:
            if arg_name not in fields:
                raise AglTypeError(
                    f"{type_label} has no field '{arg_name}'.",
                    span=provided[arg_name].span,
                )

        for field_name in fields:
            if isinstance(owner, ExceptionType) and field_name == "trace_id":
                continue
            if field_name not in provided:
                raise AglTypeError(
                    f"Missing field '{field_name}' in constructor call for "
                    f"{type_label}.",
                    span=node.span,
                )

        for arg in node.args:
            expected_field_type = fields[arg.name]
            arg_type = self._check_expr(arg.value, expected=expected_field_type)
            self._assert_assignable(arg_type, expected_field_type, arg.span)

        return owner

    # --- list / dict literals ---

    def _expected_elem_type(self, expected: Type | None) -> Type | None:
        if isinstance(expected, ListType):
            return expected.elem
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _expected_value_type(self, expected: Type | None) -> Type | None:
        if isinstance(expected, DictType):
            return expected.value
        if isinstance(expected, JsonType):
            return JsonType()
        return None

    def _check_list_lit(self, node: ListLit, *, expected: Type | None) -> Type:
        elem_expected = self._expected_elem_type(expected)
        if not node.elements:
            if elem_expected is None:
                raise AglTypeError(
                    "Empty list literal requires a type annotation "
                    "(e.g. 'let xs: list[text] = []').",
                    span=node.span,
                )
            return ListType(elem=elem_expected)
        if elem_expected is not None:
            for elem in node.elements:
                et = self._check_expr(elem, expected=elem_expected)
                self._assert_assignable(et, elem_expected, elem.span)
            return ListType(elem=elem_expected)
        unified = self._unify_elements(node.elements, kind="List", span=node.span)
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
            [entry.value for entry in node.entries], kind="Dict", span=node.span
        )
        return DictType(value=unified)

    def _unify_elements(
        self, elements: Sequence[Expr], *, kind: str, span: SourceSpan
    ) -> Type:
        """Unify literal element types with int → decimal widening."""
        types = [self._check_expr(e, expected=None) for e in elements]
        unified = types[0]
        for t in types[1:]:
            if is_assignable(unified, t):
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
            lit_type = self._check_expr(pattern.literal, expected=None)
            if not comparable_types(lit_type, subj_type):
                raise AglTypeError(
                    f"Literal pattern of type '{lit_type!r}' is incompatible with "
                    f"scrutinee of type '{subj_type!r}'.",
                    span=pattern.span,
                )
        elif isinstance(pattern, VarPattern):
            self._env.set_binding_type(pattern.node_id, subj_type)
        else:
            assert isinstance(pattern, ConstructorPattern), (
                f"Unexpected pattern kind: {type(pattern).__name__}"
            )
            if not isinstance(subj_type, EnumType):
                raise AglTypeError(
                    f"Cannot match constructor pattern '{pattern.name}' against "
                    f"non-enum type '{subj_type!r}'.",
                    span=pattern.span,
                )
            if pattern.qualifier is not None:
                self._check_variant_qualifier(pattern.qualifier, subj_type, pattern.span)
            variant_name = pattern.name
            if variant_name not in subj_type.variants:
                raise AglTypeError(
                    f"Variant '{variant_name}' does not belong to enum "
                    f"'{subj_type.name}'.",
                    span=pattern.span,
                )
            vfields = subj_type.variants[variant_name]
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
        """Emit a warning when an enum ``case`` leaves some variants uncovered."""
        if not isinstance(subj_type, EnumType):
            return
        covered: set[str] = set()
        for pattern in patterns:
            if isinstance(pattern, (WildcardPattern, VarPattern)):
                return
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
    # Branch unification
    # ------------------------------------------------------------------

    def _unify_branch_types(
        self,
        branch_types: list[Type],
        span: SourceSpan,
        construct: str,
    ) -> Type:
        """Unify branch types with int→decimal widening and BottomType filtering."""
        non_bottom = [t for t in branch_types if not isinstance(t, BottomType)]
        if not non_bottom:
            return BottomType()
        result_type = non_bottom[0]
        for bt in non_bottom[1:]:
            if bt == result_type:
                continue
            if isinstance(result_type, IntType) and isinstance(bt, DecimalType):
                result_type = DecimalType()
            elif isinstance(result_type, DecimalType) and isinstance(bt, IntType):
                pass
            else:
                raise AglTypeError(
                    f"{construct} branches have incompatible types: "
                    f"'{result_type!r}' and '{bt!r}'.",
                    span=span,
                )
        return result_type

    # ------------------------------------------------------------------
    # Bool condition helper
    # ------------------------------------------------------------------

    def _require_bool_condition(self, cond_type: Type, span: SourceSpan, kw: str) -> None:
        if not isinstance(cond_type, BoolType):
            raise AglTypeError(
                f"'{kw}' condition must be bool; got '{cond_type!r}'.",
                span=span,
            )

    # ------------------------------------------------------------------
    # Assignability helpers
    # ------------------------------------------------------------------

    def _assert_assignable(self, value_type: Type, target_type: Type, span: SourceSpan) -> None:
        if is_assignable(value_type, target_type):
            return
        raise AglTypeError(
            f"Type mismatch: expected '{target_type!r}', got '{value_type!r}'.",
            span=span,
        )

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
            function_signatures=self._env.all_function_signatures(),
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
    """Run the full type-checking pass.

    Parameters
    ----------
    resolved:
        Output of the scope resolution pass.
    capabilities:
        Immutable host capability catalog (agents, codecs, renderers).
    seed_env:
        When given, the working ``TypeEnvironment`` starts pre-populated with
        the seed's user-declared types and prior binding types.

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

    builder = _TypeBuilder(env)
    builder.collect(program)

    checker = _Checker(env=env, resolved=resolved, capabilities=capabilities)
    checker.check_program(program)

    return checker.result(resolved)
