"""``walk()`` traversal for the AgL syntax package.

``walk(node, callback)`` is a recursive traversal that calls *callback* for
every node in the tree in pre-order (parent before children). It is a
closed-set dispatcher, so adding a new node class without updating ``walk`` is
immediately visible as a ``TypeError``.

Usage::

    from agm.agl.syntax.visitor import walk

    def show(node: object) -> None:
        if isinstance(node, LetDecl):
            print(f"let {node.pattern}")

    walk(program, show)
"""

from __future__ import annotations

from collections.abc import Callable

from agm.agl.syntax.nodes import (
    ArrayLit,
    AsPattern,
    AssignStmt,
    Attribute,
    AttributeKeyedArg,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    BuiltinVarDecl,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DictEntry,
    DictLit,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    ExportItem,
    FieldAccess,
    FieldTarget,
    FuncDef,
    If,
    IfBranch,
    ImportDecl,
    ImportItem,
    IndexAccess,
    IndexTarget,
    InfixDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    LiteralPattern,
    Loop,
    NamedArg,
    NameTarget,
    NullLit,
    OperatorRef,
    Param,
    PatternField,
    Placeholder,
    Program,
    QualifierChain,
    QualifierSegment,
    Raise,
    RawInfixChain,
    RawInfixOperand,
    RawInfixOperator,
    RawPrefixNot,
    RecordDef,
    RecordUpdate,
    Return,
    ScopeRegion,
    ScopeSegment,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    UseDecl,
    VarDecl,
    VariantDef,
    VariantRef,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.types import (
    AppliedT,
    ArrayT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    NameT,
    TextT,
    UnitT,
)

# ---------------------------------------------------------------------------
# Known-node set (for loud failure on unknown types)
# ---------------------------------------------------------------------------

# NOTE: This set must stay in lockstep with walk()'s dispatch below: every node
# class listed here must have a matching ``isinstance`` branch in walk(), and
# vice versa.  Adding a node class requires updating both.
_KNOWN_NODE_TYPES: frozenset[type] = frozenset(
    {
        Program,
        # declaration attributes
        Attribute,
        AttributeKeyedArg,
        # type nodes
        TextT,
        JsonT,
        BoolT,
        IntT,
        DecimalT,
        NameT,
        ArrayT,
        DictT,
        UnitT,
        FuncT,
        AppliedT,
        # module system nodes
        QualifierSegment,
        QualifierChain,
        ImportItem,
        ImportDecl,
        ExportItem,
        ExportDecl,
        UseDecl,
        ScopeSegment,
        # declaration nodes
        RecordDef,
        VariantDef,
        VariantRef,
        EnumDef,
        ExceptionDef,
        TypeAlias,
        FuncDef,
        BuiltinVarDecl,
        InfixDecl,
        ScopeRegion,
        # binder nodes
        LetDecl,
        VarDecl,
        AssignStmt,
        NameTarget,
        IndexTarget,
        FieldTarget,
        # literal nodes
        UnitLit,
        IntLit,
        DecimalLit,
        BoolLit,
        NullLit,
        StringLit,
        ArrayLit,
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
        Placeholder,
        BinaryOp,
        OperatorRef,
        UnaryNot,
        UnaryNeg,
        Cast,
        IsTest,
        TypeApply,
        Call,
        RecordUpdate,
        Param,
        Lambda,
        Block,
        IfBranch,
        If,
        CaseBranch,
        Case,
        Loop,
        Break,
        Continue,
        CatchClause,
        Try,
        Raise,
        Return,
        RawInfixChain,
        RawInfixOperand,
        RawInfixOperator,
        RawPrefixNot,
        # pattern nodes
        WildcardPattern,
        LiteralPattern,
        VarPattern,
        AsPattern,
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


def _walk_attributes(attributes: tuple[Attribute, ...], callback: Callable[[object], None]) -> None:
    """Walk a declaration's attribute prefix before the declaration's own children."""
    for attribute in attributes:
        walk(attribute, callback)


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

    # --- Declaration attributes ---
    elif isinstance(node, Attribute):
        for attribute_arg in node.args:
            walk(attribute_arg, callback)
        for attribute_keyed in node.keyed_args:
            walk(attribute_keyed, callback)

    elif isinstance(node, AttributeKeyedArg):
        walk(node.key, callback)
        walk(node.value, callback)

    # --- Type nodes ---
    elif isinstance(node, (TextT, JsonT, BoolT, IntT, DecimalT)):
        pass  # leaves

    elif isinstance(node, NameT):
        if node.qualifier is not None:
            walk(node.qualifier, callback)

    elif isinstance(node, ArrayT):
        walk(node.elem, callback)

    elif isinstance(node, DictT):
        walk(node.value, callback)

    elif isinstance(node, UnitT):
        pass  # leaves

    elif isinstance(node, FuncT):
        for param_t in node.params:
            walk(param_t, callback)
        walk(node.result, callback)

    elif isinstance(node, AppliedT):
        if node.qualifier is not None:
            walk(node.qualifier, callback)
        for arg in node.args:
            walk(arg, callback)

    # --- Module system nodes ---
    elif isinstance(node, QualifierSegment):
        for type_arg in node.type_args or ():
            walk(type_arg, callback)

    elif isinstance(node, QualifierChain):
        for segment in node.segments:
            walk(segment, callback)

    elif isinstance(node, ImportItem):
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)

    elif isinstance(node, ImportDecl):
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for import_item in (*(node.tail or ()), *node.hidden):
            walk(import_item, callback)

    elif isinstance(node, ExportItem):
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)

    elif isinstance(node, ExportDecl):
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for export_item in (*node.items, *node.hidden):
            walk(export_item, callback)

    elif isinstance(node, UseDecl):
        for scope_segment in (*node.scope_path, *node.target):
            walk(scope_segment, callback)
        for import_item in (*(node.tail or ()), *node.hidden):
            walk(import_item, callback)

    elif isinstance(node, ScopeSegment):
        pass  # leaf — name is a plain string

    # --- Declaration nodes ---
    elif isinstance(node, RecordDef):
        _walk_attributes(node.attributes, callback)
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for f in node.fields:
            walk(f, callback)

    elif isinstance(node, VariantDef):
        _walk_attributes(node.attributes, callback)
        for f in node.fields:
            walk(f, callback)

    elif isinstance(node, VariantRef):
        walk(node.chain, callback)
        for type_arg in node.type_args:
            walk(type_arg, callback)

    elif isinstance(node, EnumDef):
        _walk_attributes(node.attributes, callback)
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for member in node.members:
            walk(member, callback)

    elif isinstance(node, ExceptionDef):
        _walk_attributes(node.attributes, callback)
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for f in node.fields:
            walk(f, callback)

    elif isinstance(node, TypeAlias):
        _walk_attributes(node.attributes, callback)
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        walk(node.type_expr, callback)

    elif isinstance(node, FuncDef):
        _walk_attributes(node.attributes, callback)
        for scope_segment in node.scope_path:
            walk(scope_segment, callback)
        for param in node.params:
            walk(param, callback)
        if node.return_type is not None:
            walk(node.return_type, callback)
        if node.body is not None:
            walk(node.body, callback)

    elif isinstance(node, BuiltinVarDecl):
        _walk_attributes(node.attributes, callback)
        walk(node.type_ann, callback)
        if node.default is not None:
            walk(node.default, callback)

    elif isinstance(node, InfixDecl):
        pass  # leaf — operator metadata only

    elif isinstance(node, ScopeRegion):
        walk(node.segment, callback)
        for scope_item in node.items:
            walk(scope_item, callback)

    # --- Binder nodes ---
    elif isinstance(node, LetDecl):
        _walk_attributes(node.attributes, callback)
        walk(node.pattern, callback)
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, VarDecl):
        _walk_attributes(node.attributes, callback)
        if node.type_ann is not None:
            walk(node.type_ann, callback)
        walk(node.value, callback)

    elif isinstance(node, AssignStmt):
        walk(node.target, callback)
        walk(node.value, callback)

    elif isinstance(node, NameTarget):
        if node.qualifier is not None:
            walk(node.qualifier, callback)

    elif isinstance(node, IndexTarget):
        walk(node.obj, callback)
        walk(node.index, callback)

    elif isinstance(node, FieldTarget):
        walk(node.obj, callback)

    # --- Literal nodes ---
    elif isinstance(node, (UnitLit, IntLit, DecimalLit, BoolLit, NullLit, StringLit)):
        pass  # leaves

    elif isinstance(node, ArrayLit):
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
        if node.qualifier is not None:
            walk(node.qualifier, callback)

    elif isinstance(node, FieldAccess):
        walk(node.obj, callback)

    elif isinstance(node, IndexAccess):
        walk(node.obj, callback)
        walk(node.index, callback)

    elif isinstance(node, NamedArg):
        walk(node.value, callback)

    elif isinstance(node, (Placeholder, OperatorRef)):
        pass

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
        if node.qualifier is not None:
            walk(node.qualifier, callback)

    elif isinstance(node, TypeApply):
        walk(node.expr, callback)
        for ta in node.type_args:
            walk(ta, callback)

    elif isinstance(node, Call):
        walk(node.callee, callback)
        for call_arg in node.args:
            walk(call_arg, callback)
        for call_named in node.named_args:
            walk(call_named, callback)
        for ta in node.type_args:
            walk(ta, callback)

    elif isinstance(node, RecordUpdate):
        walk(node.target, callback)
        for update in node.updates:
            walk(update, callback)

    elif isinstance(node, Param):
        _walk_attributes(node.attributes, callback)
        if node.type_expr is not None:
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
        for block_item in node.items:
            walk(block_item, callback)

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

    elif isinstance(node, Loop):
        if node.for_iter is not None:
            walk(node.for_iter, callback)
        if node.for_range_to is not None:
            walk(node.for_range_to, callback)
        if node.for_range_step is not None:
            walk(node.for_range_step, callback)
        if node.while_cond is not None:
            walk(node.while_cond, callback)
        if node.bound is not None:
            walk(node.bound, callback)
        walk(node.body, callback)
        if node.until_cond is not None:
            walk(node.until_cond, callback)

    elif isinstance(node, CatchClause):
        walk(node.body, callback)

    elif isinstance(node, Try):
        walk(node.body, callback)
        for clause in node.handlers:
            walk(clause, callback)

    elif isinstance(node, Raise):
        walk(node.exc, callback)

    elif isinstance(node, Return):
        if node.value is not None:
            walk(node.value, callback)

    elif isinstance(node, RawInfixChain):
        for operand in node.operands:
            walk(operand, callback)
        for operator in node.operators:
            walk(operator, callback)

    elif isinstance(node, RawInfixOperand):
        walk(node.expr, callback)
        for prefix in node.prefix_nots:
            walk(prefix, callback)

    elif isinstance(node, (RawInfixOperator, RawPrefixNot)):
        pass

    elif isinstance(node, Break | Continue):
        pass  # leaf

    # --- Pattern nodes ---
    elif isinstance(node, WildcardPattern):
        pass  # leaf

    elif isinstance(node, LiteralPattern):
        walk(node.literal, callback)

    elif isinstance(node, VarPattern):
        pass  # leaf

    elif isinstance(node, AsPattern):
        walk(node.pattern, callback)

    elif isinstance(node, PatternField):
        walk(node.pattern, callback)

    elif isinstance(node, ConstructorPattern):
        if node.qualifier is not None:
            walk(node.qualifier, callback)
        for p in node.positional:
            walk(p, callback)
        for pf in node.named:
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
