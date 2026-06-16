"""AST visitor base and ``walk()`` helper for the AgL syntax package.

Design
------
``Visitor`` is an open-dispatch base class.  ``dispatch(node)`` looks up a
method named ``visit_<ClassName>`` on the concrete subclass.  If the method is
not overridden, the default implementation is a no-op.

``walk(node, callback)`` is a standalone recursive traversal that calls
*callback* for every node in the tree in pre-order (parent before children).
It is implemented as a closed-set dispatcher so that adding a new node class
without updating ``walk`` is immediately visible as a ``TypeError``.

Usage::

    from agm.agl.syntax.visitor import Visitor, walk

    class Printer(Visitor):
        def visit_LetDecl(self, node: LetDecl) -> None:
            print(f"let {node.name}")

    printer = Printer()
    walk(program, printer.dispatch)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from agm.agl.syntax.nodes import (
    AbortPolicy,
    AgentCall,
    AgentDecl,
    BinaryOp,
    BoolLit,
    CallOptions,
    CaseExpr,
    CaseExprBranch,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DictEntry,
    DictLit,
    DoUntil,
    ElseSentinel,
    EnumDef,
    ExprStmt,
    FieldAccess,
    FieldDef,
    IfBranch,
    IfStmt,
    InterpSegment,
    IntLit,
    IsTest,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NullLit,
    ParamDecl,
    PassStmt,
    PatternField,
    PrintStmt,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    RetryPolicy,
    SetStmt,
    StringLit,
    Template,
    TextSegment,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VariantDef,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.types import (
    BoolT,
    DecimalT,
    DictT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
)

# ---------------------------------------------------------------------------
# Visitor base
# ---------------------------------------------------------------------------


class Visitor:
    """Open-dispatch visitor base class.

    Subclass and override ``visit_<NodeClassName>`` methods as needed.
    Unoverridden methods are no-ops.  Calling ``dispatch`` on an object that
    is not a known AST node raises ``TypeError``.
    """

    def dispatch(self, node: object) -> None:
        """Dispatch to the appropriate ``visit_*`` method.

        Every known AST node class has a corresponding ``visit_<ClassName>``
        method defined on this base class (defaulting to no-op).  For unknown
        types (i.e. non-AST objects) this raises ``TypeError`` so that callers
        catch accidental misuse early.
        """
        cls_name = type(node).__name__
        method_name = f"visit_{cls_name}"
        method = cast(
            "Callable[[object], None] | None",
            getattr(self, method_name, None),
        )
        if method is not None:
            method(node)
        else:
            raise TypeError(
                f"Visitor.dispatch received unknown node type {type(node)!r}. "
                "Did you forget to add a visit_ method or update the known-node set?"
            )

    # --- Default no-op visit methods (one per node class) ---

    def visit_Program(self, node: Program) -> None: ...
    def visit_TextT(self, node: TextT) -> None: ...
    def visit_JsonT(self, node: JsonT) -> None: ...
    def visit_BoolT(self, node: BoolT) -> None: ...
    def visit_IntT(self, node: IntT) -> None: ...
    def visit_DecimalT(self, node: DecimalT) -> None: ...
    def visit_NameT(self, node: NameT) -> None: ...
    def visit_ListT(self, node: ListT) -> None: ...
    def visit_DictT(self, node: DictT) -> None: ...
    def visit_FieldDef(self, node: FieldDef) -> None: ...
    def visit_RecordDef(self, node: RecordDef) -> None: ...
    def visit_VariantDef(self, node: VariantDef) -> None: ...
    def visit_EnumDef(self, node: EnumDef) -> None: ...
    def visit_TypeAlias(self, node: TypeAlias) -> None: ...
    def visit_ParamDecl(self, node: ParamDecl) -> None: ...
    def visit_ProgramDecl(self, node: ProgramDecl) -> None: ...
    def visit_AgentDecl(self, node: AgentDecl) -> None: ...
    def visit_IntLit(self, node: IntLit) -> None: ...
    def visit_DecimalLit(self, node: DecimalLit) -> None: ...
    def visit_BoolLit(self, node: BoolLit) -> None: ...
    def visit_NullLit(self, node: NullLit) -> None: ...
    def visit_StringLit(self, node: StringLit) -> None: ...
    def visit_ListLit(self, node: ListLit) -> None: ...
    def visit_DictEntry(self, node: DictEntry) -> None: ...
    def visit_DictLit(self, node: DictLit) -> None: ...
    def visit_TextSegment(self, node: TextSegment) -> None: ...
    def visit_InterpSegment(self, node: InterpSegment) -> None: ...
    def visit_Template(self, node: Template) -> None: ...
    def visit_VarRef(self, node: VarRef) -> None: ...
    def visit_FieldAccess(self, node: FieldAccess) -> None: ...
    def visit_NamedArg(self, node: NamedArg) -> None: ...
    def visit_Constructor(self, node: Constructor) -> None: ...
    def visit_AbortPolicy(self, node: AbortPolicy) -> None: ...
    def visit_RetryPolicy(self, node: RetryPolicy) -> None: ...
    def visit_CallOptions(self, node: CallOptions) -> None: ...
    def visit_AgentCall(self, node: AgentCall) -> None: ...
    def visit_BinaryOp(self, node: BinaryOp) -> None: ...
    def visit_UnaryNot(self, node: UnaryNot) -> None: ...
    def visit_UnaryNeg(self, node: UnaryNeg) -> None: ...
    def visit_IsTest(self, node: IsTest) -> None: ...
    def visit_CaseExprBranch(self, node: CaseExprBranch) -> None: ...
    def visit_CaseExpr(self, node: CaseExpr) -> None: ...
    def visit_WildcardPattern(self, node: WildcardPattern) -> None: ...
    def visit_LiteralPattern(self, node: LiteralPattern) -> None: ...
    def visit_VarPattern(self, node: VarPattern) -> None: ...
    def visit_PatternField(self, node: PatternField) -> None: ...
    def visit_ConstructorPattern(self, node: ConstructorPattern) -> None: ...
    def visit_LetDecl(self, node: LetDecl) -> None: ...
    def visit_VarDecl(self, node: VarDecl) -> None: ...
    def visit_SetStmt(self, node: SetStmt) -> None: ...
    def visit_PassStmt(self, node: PassStmt) -> None: ...
    def visit_PrintStmt(self, node: PrintStmt) -> None: ...
    def visit_ExprStmt(self, node: ExprStmt) -> None: ...
    def visit_DoUntil(self, node: DoUntil) -> None: ...
    def visit_IfBranch(self, node: IfBranch) -> None: ...
    def visit_IfStmt(self, node: IfStmt) -> None: ...
    def visit_CaseStmtBranch(self, node: CaseStmtBranch) -> None: ...
    def visit_CaseStmt(self, node: CaseStmt) -> None: ...
    def visit_CatchClause(self, node: CatchClause) -> None: ...
    def visit_TryCatch(self, node: TryCatch) -> None: ...
    def visit_Raise(self, node: Raise) -> None: ...
    def visit_ElseSentinel(self, node: ElseSentinel) -> None: ...


# ---------------------------------------------------------------------------
# Known-node set (for loud failure on unknown types)
# ---------------------------------------------------------------------------

# NOTE: This set must stay in lockstep with walk()'s dispatch below: every node
# class listed here must have a matching ``isinstance`` branch in walk(), and
# vice versa.  Adding a node class requires updating both.
_KNOWN_NODE_TYPES: frozenset[type] = frozenset(
    {
        Program,
        TextT,
        JsonT,
        BoolT,
        IntT,
        DecimalT,
        NameT,
        ListT,
        DictT,
        FieldDef,
        RecordDef,
        VariantDef,
        EnumDef,
        TypeAlias,
        ParamDecl,
        ProgramDecl,
        AgentDecl,
        IntLit,
        DecimalLit,
        BoolLit,
        NullLit,
        StringLit,
        ListLit,
        DictEntry,
        DictLit,
        TextSegment,
        InterpSegment,
        Template,
        VarRef,
        FieldAccess,
        NamedArg,
        Constructor,
        AbortPolicy,
        RetryPolicy,
        CallOptions,
        AgentCall,
        BinaryOp,
        UnaryNot,
        UnaryNeg,
        IsTest,
        CaseExprBranch,
        CaseExpr,
        WildcardPattern,
        LiteralPattern,
        VarPattern,
        PatternField,
        ConstructorPattern,
        LetDecl,
        VarDecl,
        SetStmt,
        PassStmt,
        PrintStmt,
        ExprStmt,
        DoUntil,
        IfBranch,
        IfStmt,
        CaseStmtBranch,
        CaseStmt,
        CatchClause,
        TryCatch,
        Raise,
        ElseSentinel,
    }
)


def _is_known_node(node: object) -> bool:
    return type(node) in _KNOWN_NODE_TYPES


# ---------------------------------------------------------------------------
# walk() — closed-set pre-order traversal
# ---------------------------------------------------------------------------


def walk(node: object, callback: Callable[[object], None]) -> None:
    """Pre-order traversal: call ``callback`` with *node*, then recurse.

    Raises ``TypeError`` if *node* is not a known AST node type.
    """
    if not _is_known_node(node):
        raise TypeError(
            f"walk() encountered unknown node type {type(node)!r}. "
            "Update agm.agl.syntax.visitor to handle new node classes."
        )

    callback(node)

    if isinstance(node, Program):
        for child in node.body:
            walk(child, callback)

    # --- Type nodes ---
    elif isinstance(node, (TextT, JsonT, BoolT, IntT, DecimalT, NameT)):
        pass  # leaves

    elif isinstance(node, ListT):
        walk(node.elem, callback)

    elif isinstance(node, DictT):
        walk(node.value, callback)

    # --- Declarations ---
    elif isinstance(node, FieldDef):
        walk(node.type_expr, callback)

    elif isinstance(node, RecordDef):
        for f in node.fields:
            walk(f, callback)

    elif isinstance(node, VariantDef):
        for f in node.fields:
            walk(f, callback)

    elif isinstance(node, EnumDef):
        for v in node.variants:
            walk(v, callback)

    elif isinstance(node, TypeAlias):
        walk(node.type_expr, callback)

    elif isinstance(node, ParamDecl):
        if node.annotation is not None:
            walk(node.annotation, callback)
        if node.default is not None:
            walk(node.default, callback)

    elif isinstance(node, ProgramDecl):
        pass  # leaf — name is a plain string

    elif isinstance(node, AgentDecl):
        pass  # leaf — name and runner are plain strings

    # --- Literals ---
    elif isinstance(node, (IntLit, DecimalLit, BoolLit, NullLit, StringLit)):
        pass  # leaves

    elif isinstance(node, ListLit):
        for elem in node.elements:
            walk(elem, callback)

    elif isinstance(node, DictEntry):
        walk(node.key, callback)
        walk(node.value, callback)

    elif isinstance(node, DictLit):
        for entry in node.entries:
            walk(entry, callback)

    # --- Template ---
    elif isinstance(node, TextSegment):
        pass  # leaf

    elif isinstance(node, InterpSegment):
        walk(node.expr, callback)

    elif isinstance(node, Template):
        for seg in node.segments:
            walk(seg, callback)

    # --- Expressions ---
    elif isinstance(node, VarRef):
        pass  # leaf

    elif isinstance(node, FieldAccess):
        walk(node.obj, callback)

    elif isinstance(node, NamedArg):
        walk(node.value, callback)

    elif isinstance(node, Constructor):
        for arg in node.args:
            walk(arg, callback)

    elif isinstance(node, AbortPolicy):
        pass  # leaf

    elif isinstance(node, RetryPolicy):
        pass  # leaf

    elif isinstance(node, CallOptions):
        if node.parse_policy is not None:
            walk(node.parse_policy, callback)

    elif isinstance(node, AgentCall):
        walk(node.options, callback)
        walk(node.template, callback)

    elif isinstance(node, BinaryOp):
        walk(node.left, callback)
        walk(node.right, callback)

    elif isinstance(node, UnaryNot):
        walk(node.operand, callback)

    elif isinstance(node, UnaryNeg):
        walk(node.operand, callback)

    elif isinstance(node, IsTest):
        walk(node.expr, callback)

    elif isinstance(node, CaseExprBranch):
        walk(node.pattern, callback)
        walk(node.body, callback)

    elif isinstance(node, CaseExpr):
        walk(node.subject, callback)
        for branch in node.branches:
            walk(branch, callback)

    # --- Patterns ---
    elif isinstance(node, WildcardPattern):
        pass  # leaf

    elif isinstance(node, LiteralPattern):
        walk(node.literal, callback)

    elif isinstance(node, VarPattern):
        pass  # leaf

    elif isinstance(node, PatternField):
        walk(node.pattern, callback)

    elif isinstance(node, ConstructorPattern):
        for pf in node.fields:
            walk(pf, callback)

    # --- Statements ---
    elif isinstance(node, LetDecl):
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, VarDecl):
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, SetStmt):
        walk(node.value, callback)

    elif isinstance(node, PassStmt):
        pass  # leaf

    elif isinstance(node, PrintStmt):
        walk(node.value, callback)

    elif isinstance(node, ExprStmt):
        walk(node.expr, callback)

    elif isinstance(node, DoUntil):
        for stmt in node.body:
            walk(stmt, callback)
        walk(node.condition, callback)

    elif isinstance(node, IfBranch):
        walk(node.cond, callback)
        for stmt in node.body:
            walk(stmt, callback)

    elif isinstance(node, IfStmt):
        for if_branch in node.branches:
            walk(if_branch, callback)

    elif isinstance(node, CaseStmtBranch):
        walk(node.pattern, callback)
        for stmt in node.body:
            walk(stmt, callback)

    elif isinstance(node, CaseStmt):
        walk(node.subject, callback)
        for case_branch in node.branches:
            walk(case_branch, callback)

    elif isinstance(node, CatchClause):
        for stmt in node.body:
            walk(stmt, callback)

    elif isinstance(node, TryCatch):
        for stmt in node.body:
            walk(stmt, callback)
        for clause in node.handlers:
            walk(clause, callback)

    elif isinstance(node, Raise):
        walk(node.exc, callback)

    else:
        # The _is_known_node guard at the top guarantees this is ElseSentinel.
        # This dispatch must stay in lockstep with _KNOWN_NODE_TYPES above:
        # every known node class needs a branch here, and vice versa.
        pass  # leaf sentinel
