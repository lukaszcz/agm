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
    AgentDecl,
    AssignStmt,
    BinaryOp,
    Block,
    BoolLit,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConfigPragma,
    ConstructorPattern,
    DecimalLit,
    DictEntry,
    DictLit,
    Do,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    FieldAccess,
    FieldDef,
    FuncDef,
    If,
    IfBranch,
    ImportDecl,
    ImportItem,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NameTarget,
    NullLit,
    Param,
    ParamDecl,
    PatternField,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VariantDef,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.types import (
    AgentT,
    AppliedT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    ListT,
    NameT,
    Qualifier,
    TextT,
    UnitT,
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

    # Type nodes
    def visit_TextT(self, node: TextT) -> None: ...
    def visit_JsonT(self, node: JsonT) -> None: ...
    def visit_BoolT(self, node: BoolT) -> None: ...
    def visit_IntT(self, node: IntT) -> None: ...
    def visit_DecimalT(self, node: DecimalT) -> None: ...
    def visit_NameT(self, node: NameT) -> None: ...
    def visit_ListT(self, node: ListT) -> None: ...
    def visit_DictT(self, node: DictT) -> None: ...
    def visit_UnitT(self, node: UnitT) -> None: ...
    def visit_AgentT(self, node: AgentT) -> None: ...
    def visit_FuncT(self, node: FuncT) -> None: ...
    def visit_AppliedT(self, node: AppliedT) -> None: ...

    # Module system nodes
    def visit_Qualifier(self, node: Qualifier) -> None: ...
    def visit_ImportItem(self, node: ImportItem) -> None: ...
    def visit_ImportDecl(self, node: ImportDecl) -> None: ...

    # Declaration nodes
    def visit_FieldDef(self, node: FieldDef) -> None: ...
    def visit_RecordDef(self, node: RecordDef) -> None: ...
    def visit_VariantDef(self, node: VariantDef) -> None: ...
    def visit_EnumDef(self, node: EnumDef) -> None: ...
    def visit_ExceptionDef(self, node: ExceptionDef) -> None: ...
    def visit_TypeAlias(self, node: TypeAlias) -> None: ...
    def visit_ParamDecl(self, node: ParamDecl) -> None: ...
    def visit_ProgramDecl(self, node: ProgramDecl) -> None: ...
    def visit_AgentDecl(self, node: AgentDecl) -> None: ...
    def visit_FuncDef(self, node: FuncDef) -> None: ...
    def visit_ConfigPragma(self, node: ConfigPragma) -> None: ...

    # Binder nodes
    def visit_LetDecl(self, node: LetDecl) -> None: ...
    def visit_VarDecl(self, node: VarDecl) -> None: ...
    def visit_AssignStmt(self, node: AssignStmt) -> None: ...
    def visit_NameTarget(self, node: NameTarget) -> None: ...
    def visit_IndexTarget(self, node: IndexTarget) -> None: ...

    # Literal nodes
    def visit_UnitLit(self, node: UnitLit) -> None: ...
    def visit_IntLit(self, node: IntLit) -> None: ...
    def visit_DecimalLit(self, node: DecimalLit) -> None: ...
    def visit_BoolLit(self, node: BoolLit) -> None: ...
    def visit_NullLit(self, node: NullLit) -> None: ...
    def visit_StringLit(self, node: StringLit) -> None: ...
    def visit_ListLit(self, node: ListLit) -> None: ...
    def visit_DictEntry(self, node: DictEntry) -> None: ...
    def visit_DictLit(self, node: DictLit) -> None: ...

    # Template nodes
    def visit_TextSegment(self, node: TextSegment) -> None: ...
    def visit_InterpSegment(self, node: InterpSegment) -> None: ...
    def visit_Template(self, node: Template) -> None: ...

    # Expression nodes
    def visit_VarRef(self, node: VarRef) -> None: ...
    def visit_FieldAccess(self, node: FieldAccess) -> None: ...
    def visit_IndexAccess(self, node: IndexAccess) -> None: ...
    def visit_NamedArg(self, node: NamedArg) -> None: ...
    def visit_BinaryOp(self, node: BinaryOp) -> None: ...
    def visit_UnaryNot(self, node: UnaryNot) -> None: ...
    def visit_UnaryNeg(self, node: UnaryNeg) -> None: ...
    def visit_Cast(self, node: Cast) -> None: ...
    def visit_IsTest(self, node: IsTest) -> None: ...
    def visit_Call(self, node: Call) -> None: ...
    def visit_Param(self, node: Param) -> None: ...
    def visit_Lambda(self, node: Lambda) -> None: ...
    def visit_Block(self, node: Block) -> None: ...
    def visit_IfBranch(self, node: IfBranch) -> None: ...
    def visit_If(self, node: If) -> None: ...
    def visit_CaseBranch(self, node: CaseBranch) -> None: ...
    def visit_Case(self, node: Case) -> None: ...
    def visit_Do(self, node: Do) -> None: ...
    def visit_CatchClause(self, node: CatchClause) -> None: ...
    def visit_Try(self, node: Try) -> None: ...
    def visit_Raise(self, node: Raise) -> None: ...

    # Pattern nodes
    def visit_WildcardPattern(self, node: WildcardPattern) -> None: ...
    def visit_LiteralPattern(self, node: LiteralPattern) -> None: ...
    def visit_VarPattern(self, node: VarPattern) -> None: ...
    def visit_PatternField(self, node: PatternField) -> None: ...
    def visit_ConstructorPattern(self, node: ConstructorPattern) -> None: ...

    # Sentinel
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
        # type nodes
        TextT,
        JsonT,
        BoolT,
        IntT,
        DecimalT,
        NameT,
        ListT,
        DictT,
        UnitT,
        AgentT,
        FuncT,
        AppliedT,
        # module system nodes
        Qualifier,
        ImportItem,
        ImportDecl,
        # declaration nodes
        FieldDef,
        RecordDef,
        VariantDef,
        EnumDef,
        ExceptionDef,
        TypeAlias,
        ParamDecl,
        ProgramDecl,
        AgentDecl,
        FuncDef,
        ConfigPragma,
        # binder nodes
        LetDecl,
        VarDecl,
        AssignStmt,
        NameTarget,
        IndexTarget,
        # literal nodes
        UnitLit,
        IntLit,
        DecimalLit,
        BoolLit,
        NullLit,
        StringLit,
        ListLit,
        DictEntry,
        DictLit,
        # template nodes
        TextSegment,
        InterpSegment,
        Template,
        # expression nodes
        VarRef,
        FieldAccess,
        IndexAccess,
        NamedArg,
        BinaryOp,
        UnaryNot,
        UnaryNeg,
        Cast,
        IsTest,
        Call,
        Param,
        Lambda,
        Block,
        IfBranch,
        If,
        CaseBranch,
        Case,
        Do,
        CatchClause,
        Try,
        Raise,
        # pattern nodes
        WildcardPattern,
        LiteralPattern,
        VarPattern,
        PatternField,
        ConstructorPattern,
        # sentinel
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
        walk(node.body, callback)

    # --- Type nodes ---
    elif isinstance(node, (TextT, JsonT, BoolT, IntT, DecimalT)):
        pass  # leaves

    elif isinstance(node, NameT):
        if node.module_qualifier is not None:
            walk(node.module_qualifier, callback)

    elif isinstance(node, ListT):
        walk(node.elem, callback)

    elif isinstance(node, DictT):
        walk(node.value, callback)

    elif isinstance(node, (UnitT, AgentT)):
        pass  # leaves

    elif isinstance(node, FuncT):
        for param_t in node.params:
            walk(param_t, callback)
        walk(node.result, callback)

    elif isinstance(node, AppliedT):
        if node.module_qualifier is not None:
            walk(node.module_qualifier, callback)
        for arg in node.args:
            walk(arg, callback)

    # --- Module system nodes ---
    elif isinstance(node, Qualifier):
        pass  # leaf — segments are plain strings

    elif isinstance(node, ImportItem):
        pass  # leaf — name and rename are plain strings

    elif isinstance(node, ImportDecl):
        for import_item in node.items:
            walk(import_item, callback)

    # --- Declaration nodes ---
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

    elif isinstance(node, ExceptionDef):
        for f in node.fields:
            walk(f, callback)

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

    elif isinstance(node, FuncDef):
        for param in node.params:
            walk(param, callback)
        walk(node.return_type, callback)
        if node.body is not None:
            walk(node.body, callback)

    elif isinstance(node, ConfigPragma):
        pass  # leaf — key and value are plain scalars

    # --- Binder nodes ---
    elif isinstance(node, LetDecl):
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, VarDecl):
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, AssignStmt):
        walk(node.target, callback)
        walk(node.value, callback)

    elif isinstance(node, NameTarget):
        pass

    elif isinstance(node, IndexTarget):
        walk(node.obj, callback)
        walk(node.index, callback)

    # --- Literal nodes ---
    elif isinstance(node, (UnitLit, IntLit, DecimalLit, BoolLit, NullLit, StringLit)):
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

    # --- Template nodes ---
    elif isinstance(node, TextSegment):
        pass  # leaf

    elif isinstance(node, InterpSegment):
        walk(node.expr, callback)

    elif isinstance(node, Template):
        for seg in node.segments:
            walk(seg, callback)

    # --- Expression nodes ---
    elif isinstance(node, VarRef):
        if node.module_qualifier is not None:
            walk(node.module_qualifier, callback)

    elif isinstance(node, FieldAccess):
        walk(node.obj, callback)

    elif isinstance(node, IndexAccess):
        walk(node.obj, callback)
        walk(node.index, callback)

    elif isinstance(node, NamedArg):
        walk(node.value, callback)

    elif isinstance(node, BinaryOp):
        walk(node.left, callback)
        walk(node.right, callback)

    elif isinstance(node, UnaryNot):
        walk(node.operand, callback)

    elif isinstance(node, UnaryNeg):
        walk(node.operand, callback)

    elif isinstance(node, Cast):
        walk(node.expr, callback)
        walk(node.target_type, callback)

    elif isinstance(node, IsTest):
        walk(node.expr, callback)

    elif isinstance(node, Call):
        walk(node.callee, callback)
        for call_arg in node.args:
            walk(call_arg, callback)
        for call_named in node.named_args:
            walk(call_named, callback)
        for ta in node.type_args:
            walk(ta, callback)

    elif isinstance(node, Param):
        walk(node.type_expr, callback)
        if node.default is not None:
            walk(node.default, callback)

    elif isinstance(node, Lambda):
        for param in node.params:
            walk(param, callback)
        if node.return_type is not None:
            walk(node.return_type, callback)
        walk(node.body, callback)

    elif isinstance(node, Block):
        for item in node.items:
            walk(item, callback)

    elif isinstance(node, IfBranch):
        walk(node.cond, callback)
        walk(node.body, callback)

    elif isinstance(node, If):
        for if_branch in node.branches:
            walk(if_branch, callback)

    elif isinstance(node, CaseBranch):
        walk(node.pattern, callback)
        walk(node.body, callback)

    elif isinstance(node, Case):
        walk(node.subject, callback)
        for case_branch in node.branches:
            walk(case_branch, callback)

    elif isinstance(node, Do):
        walk(node.body, callback)
        walk(node.condition, callback)

    elif isinstance(node, CatchClause):
        walk(node.body, callback)

    elif isinstance(node, Try):
        walk(node.body, callback)
        for clause in node.handlers:
            walk(clause, callback)

    elif isinstance(node, Raise):
        walk(node.exc, callback)

    # --- Pattern nodes ---
    elif isinstance(node, WildcardPattern):
        pass  # leaf

    elif isinstance(node, LiteralPattern):
        walk(node.literal, callback)

    elif isinstance(node, VarPattern):
        pass  # leaf

    elif isinstance(node, PatternField):
        walk(node.pattern, callback)

    elif isinstance(node, ConstructorPattern):
        if node.module_qualifier is not None:
            walk(node.module_qualifier, callback)
        for pf in node.fields:
            walk(pf, callback)

    elif isinstance(node, ElseSentinel):
        pass  # leaf sentinel

    else:
        # A node is in _KNOWN_NODE_TYPES (the guard at the top passed) but has no
        # walk branch here.  This dispatch MUST stay in lockstep with
        # _KNOWN_NODE_TYPES: every known node class needs an explicit branch.
        # Fail loudly rather than silently dropping the node's children.
        raise AssertionError(
            f"walk(): node type {type(node)!r} is known but has no walk branch. "
            "Add an isinstance branch (and keep _KNOWN_NODE_TYPES in lockstep)."
        )
