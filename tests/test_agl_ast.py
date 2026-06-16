"""Tests for the AgL AST package (agm.agl.syntax).

Covers:
- SourceSpan construction and move from diagnostics to syntax.spans
- TypeExpr hierarchy (TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT)
- All AST node types: Program, declarations, statements, expressions, patterns
- Equality semantics: equal structure with different spans/node_ids compare equal
- Immutability: frozen dataclasses reject mutation
- Visitor/walk traversal visits every node kind
- Tuple-typed children (not lists)
- ELSE sentinel type for IfBranch.cond
- CallOptions and ParsePolicy (AbortPolicy, RetryPolicy)
- BinaryOp with closed operator set
- DecimalLit holds decimal.Decimal
"""

from __future__ import annotations

import decimal
from dataclasses import FrozenInstanceError

import pytest

# SourceSpan should also still be importable from diagnostics
from agm.agl.diagnostics import SourceSpan as DiagnosticsSourceSpan

# ---------------------------------------------------------------------------
# Import everything through the package public API
# ---------------------------------------------------------------------------
from agm.agl.syntax import (
    # sentinel
    ELSE,
    AbortPolicy,
    AgentCall,
    AgentDecl,
    BinaryOp,
    BinOp,
    BoolLit,
    BoolT,
    # nodes – call options
    CallOptions,
    CaseExpr,
    CaseExprBranch,
    CaseStmt,
    CaseStmtBranch,
    CatchClause,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DecimalT,
    DictEntry,
    DictLit,
    DictT,
    DoUntil,
    ElseSentinel,
    EnumDef,
    Expr,
    ExprStmt,
    FieldAccess,
    FieldDef,
    IfBranch,
    IfExpr,
    IfExprBranch,
    IfStmt,
    InputDecl,
    InterpSegment,
    IntLit,
    IntT,
    IsTest,
    JsonT,
    # nodes – statements
    LetDecl,
    ListLit,
    ListT,
    LiteralPattern,
    NamedArg,
    NameT,
    NullLit,
    PassStmt,
    Pattern,
    PatternField,
    PrintStmt,
    # nodes – program
    Program,
    Raise,
    # nodes – declarations
    RecordDef,
    RetryPolicy,
    SetStmt,
    # spans
    SourceSpan,
    # union aliases
    Stmt,
    StringLit,
    Template,
    TemplateSegment,
    TextSegment,
    # types
    TextT,
    TryCatch,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    VarDecl,
    VariantDef,
    VarPattern,
    # nodes – expressions
    VarRef,
    # nodes – patterns
    WildcardPattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def span(sl: int = 1, sc: int = 0, el: int = 1, ec: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=sl, start_col=sc, end_line=el, end_col=ec,
        start_offset=0, end_offset=1,
    )


def nid(n: int = 1) -> int:
    return n


# ---------------------------------------------------------------------------
# SourceSpan
# ---------------------------------------------------------------------------

class TestSourceSpan:
    def test_construction_positional_fields(self) -> None:
        s = SourceSpan(
            start_line=1, start_col=0, end_line=1, end_col=10,
            start_offset=0, end_offset=10,
        )
        assert s.start_line == 1
        assert s.start_col == 0
        assert s.end_line == 1
        assert s.end_col == 10
        assert s.start_offset == 0
        assert s.end_offset == 10

    def test_equality(self) -> None:
        s1 = SourceSpan(1, 0, 1, 10, 0, 10)
        s2 = SourceSpan(1, 0, 1, 10, 0, 10)
        assert s1 == s2

    def test_frozen(self) -> None:
        s = SourceSpan(1, 0, 1, 5, 0, 5)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(s, "start_line", 99)

    def test_diagnostics_import_same_class(self) -> None:
        # The lexer imports SourceSpan from diagnostics; it must be the same class.
        assert SourceSpan is DiagnosticsSourceSpan

    def test_offsets_are_required(self) -> None:
        # Offsets are mandatory: a 4-arg construction is an argument error.
        args = (1, 1, 1, 10)
        with pytest.raises(TypeError):
            SourceSpan(*args)


# ---------------------------------------------------------------------------
# TypeExpr hierarchy
# ---------------------------------------------------------------------------

class TestTypeExprs:
    def _s(self) -> SourceSpan:
        return span()

    def test_text_t(self) -> None:
        t = TextT(span=self._s(), node_id=1)
        assert isinstance(t, TextT)

    def test_json_t(self) -> None:
        t = JsonT(span=self._s(), node_id=1)
        assert isinstance(t, JsonT)

    def test_bool_t(self) -> None:
        t = BoolT(span=self._s(), node_id=1)
        assert isinstance(t, BoolT)

    def test_int_t(self) -> None:
        t = IntT(span=self._s(), node_id=1)
        assert isinstance(t, IntT)

    def test_decimal_t(self) -> None:
        t = DecimalT(span=self._s(), node_id=1)
        assert isinstance(t, DecimalT)

    def test_name_t(self) -> None:
        t = NameT(name="MyType", span=self._s(), node_id=1)
        assert t.name == "MyType"

    def test_list_t(self) -> None:
        elem = TextT(span=self._s(), node_id=2)
        t = ListT(elem=elem, span=self._s(), node_id=1)
        assert t.elem is elem

    def test_dict_t(self) -> None:
        val = IntT(span=self._s(), node_id=2)
        t = DictT(value=val, span=self._s(), node_id=1)
        assert t.value is val

    def test_type_equality_ignores_span_and_node_id(self) -> None:
        s1 = span(1, 0, 1, 5)
        s2 = span(2, 0, 2, 5)
        t1 = TextT(span=s1, node_id=1)
        t2 = TextT(span=s2, node_id=99)
        assert t1 == t2

    def test_name_t_equality(self) -> None:
        t1 = NameT(name="Foo", span=span(1, 0, 1, 3), node_id=1)
        t2 = NameT(name="Foo", span=span(2, 0, 2, 3), node_id=99)
        assert t1 == t2

    def test_name_t_inequality(self) -> None:
        t1 = NameT(name="Foo", span=span(), node_id=1)
        t2 = NameT(name="Bar", span=span(), node_id=1)
        assert t1 != t2

    def test_list_t_equality(self) -> None:
        a = ListT(elem=TextT(span=span(), node_id=2), span=span(), node_id=1)
        b = ListT(elem=TextT(span=span(3, 0, 3, 4), node_id=99), span=span(5, 0, 5, 4), node_id=50)
        assert a == b

    def test_type_frozen(self) -> None:
        t = TextT(span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(t, "span", span())


# ---------------------------------------------------------------------------
# Literal nodes
# ---------------------------------------------------------------------------

class TestLiterals:
    def _s(self) -> SourceSpan:
        return span()

    def test_int_lit(self) -> None:
        node = IntLit(value=42, span=self._s(), node_id=1)
        assert node.value == 42

    def test_decimal_lit(self) -> None:
        d = decimal.Decimal("3.14")
        node = DecimalLit(value=d, span=self._s(), node_id=1)
        assert node.value == d

    def test_decimal_lit_holds_decimal_type(self) -> None:
        node = DecimalLit(value=decimal.Decimal("1.0"), span=self._s(), node_id=1)
        assert isinstance(node.value, decimal.Decimal)

    def test_bool_lit_true(self) -> None:
        node = BoolLit(value=True, span=self._s(), node_id=1)
        assert node.value is True

    def test_bool_lit_false(self) -> None:
        node = BoolLit(value=False, span=self._s(), node_id=1)
        assert node.value is False

    def test_null_lit(self) -> None:
        node = NullLit(span=self._s(), node_id=1)
        assert isinstance(node, NullLit)

    def test_string_lit(self) -> None:
        node = StringLit(value="hello", span=self._s(), node_id=1)
        assert node.value == "hello"

    def test_list_lit(self) -> None:
        elems: tuple[Expr, ...] = (IntLit(value=1, span=self._s(), node_id=2),)
        node = ListLit(elements=elems, span=self._s(), node_id=1)
        assert isinstance(node.elements, tuple)
        assert len(node.elements) == 1

    def test_list_lit_empty(self) -> None:
        node = ListLit(elements=(), span=self._s(), node_id=1)
        assert node.elements == ()

    def test_dict_lit(self) -> None:
        entry = DictEntry(
            key=StringLit(value="k", span=self._s(), node_id=3),
            value=IntLit(value=1, span=self._s(), node_id=4),
            span=self._s(),
            node_id=2,
        )
        node = DictLit(entries=((entry,)), span=self._s(), node_id=1)
        assert isinstance(node.entries, tuple)

    def test_equality_ignores_span_node_id(self) -> None:
        a = IntLit(value=5, span=span(1, 0, 1, 1), node_id=1)
        b = IntLit(value=5, span=span(9, 3, 9, 4), node_id=999)
        assert a == b

    def test_inequality_different_value(self) -> None:
        a = IntLit(value=1, span=span(), node_id=1)
        b = IntLit(value=2, span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# Template nodes
# ---------------------------------------------------------------------------

class TestTemplate:
    def _s(self) -> SourceSpan:
        return span()

    def test_text_segment(self) -> None:
        seg = TextSegment(text="hello", span=self._s(), node_id=1)
        assert seg.text == "hello"

    def test_interp_segment_no_render(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        seg = InterpSegment(expr=expr, span=self._s(), node_id=1)
        assert seg.expr is expr

    def test_template_segments_are_tuple(self) -> None:
        segs: tuple[TemplateSegment, ...] = (
            TextSegment(text="hello ", span=self._s(), node_id=2),
            InterpSegment(
                expr=VarRef(name="name", span=self._s(), node_id=3),
                span=self._s(),
                node_id=4,
            ),
        )
        tmpl = Template(segments=segs, span=self._s(), node_id=1)
        assert isinstance(tmpl.segments, tuple)
        assert len(tmpl.segments) == 2


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

class TestExpressions:
    def _s(self) -> SourceSpan:
        return span()

    def test_var_ref(self) -> None:
        node = VarRef(name="x", span=self._s(), node_id=1)
        assert node.name == "x"

    def test_field_access(self) -> None:
        obj = VarRef(name="obj", span=self._s(), node_id=2)
        node = FieldAccess(obj=obj, field="attr", span=self._s(), node_id=1)
        assert node.field == "attr"
        assert node.obj is obj

    def test_agent_call_minimal(self) -> None:
        tmpl = Template(segments=(), span=self._s(), node_id=3)
        opts = CallOptions(
            format=None, strict_json=None, parse_policy=None, span=self._s(), node_id=2
        )
        node = AgentCall(agent="ask", options=opts, template=tmpl, span=self._s(), node_id=1)
        assert node.agent == "ask"

    def test_constructor(self) -> None:
        arg = NamedArg(
            name="field",
            value=IntLit(value=1, span=self._s(), node_id=3),
            span=self._s(),
            node_id=2,
        )
        node = Constructor(qualifier=None, name="MyType", args=(arg,), span=self._s(), node_id=1)
        assert isinstance(node.args, tuple)
        assert node.qualifier is None

    def test_constructor_with_qualifier(self) -> None:
        node = Constructor(
            qualifier="Ns",
            name="Variant",
            args=(),
            span=self._s(),
            node_id=1,
        )
        assert node.qualifier == "Ns"
        assert node.name == "Variant"

    def test_binary_op(self) -> None:
        left = IntLit(value=1, span=self._s(), node_id=2)
        right = IntLit(value=2, span=self._s(), node_id=3)
        node = BinaryOp(op=BinOp.ADD, left=left, right=right, span=self._s(), node_id=1)
        assert node.op is BinOp.ADD

    def test_all_binary_ops_exist(self) -> None:
        ops = {BinOp.EQ, BinOp.NEQ, BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE,
               BinOp.IN, BinOp.AND, BinOp.OR, BinOp.ADD, BinOp.SUB,
               BinOp.MUL, BinOp.DIV}
        assert len(ops) == 13

    def test_unary_not(self) -> None:
        operand = BoolLit(value=True, span=self._s(), node_id=2)
        node = UnaryNot(operand=operand, span=self._s(), node_id=1)
        assert node.operand is operand

    def test_unary_neg(self) -> None:
        operand = IntLit(value=5, span=self._s(), node_id=2)
        node = UnaryNeg(operand=operand, span=self._s(), node_id=1)
        assert node.operand is operand

    def test_is_test(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        node = IsTest(
            expr=expr, qualifier=None, variant="Some", negated=False, span=self._s(), node_id=1
        )
        assert node.negated is False
        assert node.variant == "Some"

    def test_is_test_negated(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        node = IsTest(
            expr=expr, qualifier=None, variant="None_", negated=True, span=self._s(), node_id=1
        )
        assert node.negated is True

    def test_case_expr(self) -> None:
        subject = VarRef(name="x", span=self._s(), node_id=2)
        pat = WildcardPattern(span=self._s(), node_id=4)
        body = NullLit(span=self._s(), node_id=5)
        branch = CaseExprBranch(pattern=pat, body=body, span=self._s(), node_id=3)
        node = CaseExpr(subject=subject, branches=(branch,), span=self._s(), node_id=1)
        assert isinstance(node.branches, tuple)

    def test_equality_ignores_span_node_id(self) -> None:
        a = VarRef(name="x", span=span(1, 0, 1, 1), node_id=10)
        b = VarRef(name="x", span=span(5, 3, 5, 4), node_id=99)
        assert a == b

    def test_inequality_different_name(self) -> None:
        a = VarRef(name="x", span=span(), node_id=1)
        b = VarRef(name="y", span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# CallOptions and ParsePolicy
# ---------------------------------------------------------------------------

class TestCallOptions:
    def _s(self) -> SourceSpan:
        return span()

    def test_call_options_defaults(self) -> None:
        opts = CallOptions(
            format=None, strict_json=None, parse_policy=None, span=self._s(), node_id=1
        )
        assert opts.format is None
        assert opts.strict_json is None
        assert opts.parse_policy is None

    def test_abort_policy(self) -> None:
        policy = AbortPolicy(span=self._s(), node_id=1)
        assert isinstance(policy, AbortPolicy)

    def test_retry_policy(self) -> None:
        policy = RetryPolicy(extra=3, span=self._s(), node_id=1)
        assert policy.extra == 3

    def test_call_options_with_all_fields(self) -> None:
        policy = RetryPolicy(extra=2, span=self._s(), node_id=2)
        opts = CallOptions(
            format="json",
            strict_json=True,
            parse_policy=policy,
            span=self._s(),
            node_id=1,
        )
        assert opts.format == "json"
        assert opts.strict_json is True
        assert opts.parse_policy is policy


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------

class TestStatements:
    def _s(self) -> SourceSpan:
        return span()

    def test_let_decl(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=2)
        node = LetDecl(name="x", type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.name == "x"
        assert node.type_ann is None

    def test_let_decl_with_type(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=3)
        t = IntT(span=self._s(), node_id=2)
        node = LetDecl(name="x", type_ann=t, value=val, span=self._s(), node_id=1)
        assert node.type_ann is t

    def test_var_decl(self) -> None:
        val = IntLit(value=0, span=self._s(), node_id=2)
        node = VarDecl(name="count", type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.name == "count"

    def test_set_stmt(self) -> None:
        val = IntLit(value=5, span=self._s(), node_id=2)
        node = SetStmt(target="count", value=val, span=self._s(), node_id=1)
        assert node.target == "count"

    def test_pass_stmt(self) -> None:
        node = PassStmt(span=self._s(), node_id=1)
        assert isinstance(node, PassStmt)

    def test_print_stmt(self) -> None:
        val = StringLit(value="hi", span=self._s(), node_id=2)
        node = PrintStmt(value=val, span=self._s(), node_id=1)
        assert node.value is val

    def test_expr_stmt(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        node = ExprStmt(expr=expr, span=self._s(), node_id=1)
        assert node.expr is expr

    def test_do_until(self) -> None:
        cond = BoolLit(value=True, span=self._s(), node_id=3)
        body: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=4),)
        node = DoUntil(limit=None, body=body, condition=cond, span=self._s(), node_id=1)
        assert isinstance(node.body, tuple)
        assert node.limit is None

    def test_do_until_with_limit(self) -> None:
        cond = BoolLit(value=True, span=self._s(), node_id=3)
        node = DoUntil(limit=10, body=(), condition=cond, span=self._s(), node_id=1)
        assert node.limit == 10

    def test_if_stmt(self) -> None:
        cond = BoolLit(value=True, span=self._s(), node_id=3)
        body: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=4),)
        branch = IfBranch(cond=cond, body=body, span=self._s(), node_id=2)
        node = IfStmt(branches=(branch,), span=self._s(), node_id=1)
        assert isinstance(node.branches, tuple)

    def test_if_branch_else_sentinel(self) -> None:
        body: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=3),)
        branch = IfBranch(cond=ELSE, body=body, span=self._s(), node_id=2)
        assert branch.cond is ELSE
        assert isinstance(branch.cond, ElseSentinel)

    def test_case_stmt(self) -> None:
        subject = VarRef(name="x", span=self._s(), node_id=2)
        pat = WildcardPattern(span=self._s(), node_id=4)
        body: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=5),)
        branch = CaseStmtBranch(pattern=pat, body=body, span=self._s(), node_id=3)
        node = CaseStmt(subject=subject, branches=(branch,), span=self._s(), node_id=1)
        assert isinstance(node.branches, tuple)

    def test_try_catch(self) -> None:
        body: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=3),)
        handler: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=5),)
        clause = CatchClause(
            exc_type=None, binding=None, body=handler, span=self._s(), node_id=4
        )
        node = TryCatch(body=body, handlers=(clause,), span=self._s(), node_id=1)
        assert isinstance(node.handlers, tuple)

    def test_catch_clause_with_type_and_binding(self) -> None:
        handler: tuple[Stmt, ...] = (PassStmt(span=self._s(), node_id=3),)
        clause = CatchClause(
            exc_type="MyError",
            binding="e",
            body=handler,
            span=self._s(),
            node_id=1,
        )
        assert clause.exc_type == "MyError"
        assert clause.binding == "e"

    def test_raise_stmt(self) -> None:
        expr = Constructor(qualifier=None, name="MyErr", args=(), span=self._s(), node_id=2)
        node = Raise(exc=expr, span=self._s(), node_id=1)
        assert node.exc is expr

    def test_stmt_equality_ignores_span_node_id(self) -> None:
        a = PassStmt(span=span(1, 0, 1, 4), node_id=1)
        b = PassStmt(span=span(9, 0, 9, 4), node_id=99)
        assert a == b

    def test_stmt_frozen(self) -> None:
        node = PassStmt(span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "span", span())


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------

class TestDeclarations:
    def _s(self) -> SourceSpan:
        return span()

    def test_field_def(self) -> None:
        t = TextT(span=self._s(), node_id=2)
        f = FieldDef(name="title", type_expr=t, span=self._s(), node_id=1)
        assert f.name == "title"
        assert f.type_expr is t

    def test_record_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = FieldDef(name="x", type_expr=t, span=self._s(), node_id=2)
        node = RecordDef(name="Point", fields=(f,), span=self._s(), node_id=1)
        assert node.name == "Point"
        assert isinstance(node.fields, tuple)

    def test_variant_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = FieldDef(name="val", type_expr=t, span=self._s(), node_id=2)
        v = VariantDef(name="Some", fields=(f,), span=self._s(), node_id=1)
        assert v.name == "Some"

    def test_enum_def(self) -> None:
        v1 = VariantDef(name="A", fields=(), span=self._s(), node_id=2)
        v2 = VariantDef(name="B", fields=(), span=self._s(), node_id=3)
        node = EnumDef(name="Color", variants=(v1, v2), span=self._s(), node_id=1)
        assert isinstance(node.variants, tuple)
        assert len(node.variants) == 2

    def test_type_alias(self) -> None:
        t = ListT(elem=TextT(span=self._s(), node_id=3), span=self._s(), node_id=2)
        node = TypeAlias(name="Names", type_expr=t, span=self._s(), node_id=1)
        assert node.name == "Names"

    def test_input_decl_annotated(self) -> None:
        t = TextT(span=self._s(), node_id=2)
        node = InputDecl(name="spec", annotation=t, span=self._s(), node_id=1)
        assert node.name == "spec"
        assert node.annotation is t

    def test_input_decl_unannotated(self) -> None:
        node = InputDecl(name="spec", annotation=None, span=self._s(), node_id=1)
        assert node.name == "spec"
        assert node.annotation is None

    def test_agent_decl_bare(self) -> None:
        node = AgentDecl(name="reviewer", runner=None, span=self._s(), node_id=1)
        assert node.name == "reviewer"
        assert node.runner is None

    def test_agent_decl_with_runner(self) -> None:
        node = AgentDecl(name="impl", runner="claude -p", span=self._s(), node_id=1)
        assert node.name == "impl"
        assert node.runner == "claude -p"

    def test_agent_decl_equality_ignores_span_and_node_id(self) -> None:
        a = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=1)
        b = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=99)
        assert a == b

    def test_agent_decl_inequality_on_runner(self) -> None:
        a = AgentDecl(name="impl", runner=None, span=span(), node_id=1)
        b = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# Pattern nodes
# ---------------------------------------------------------------------------

class TestPatterns:
    def _s(self) -> SourceSpan:
        return span()

    def test_wildcard_pattern(self) -> None:
        p = WildcardPattern(span=self._s(), node_id=1)
        assert isinstance(p, WildcardPattern)

    def test_literal_pattern(self) -> None:
        lit = IntLit(value=42, span=self._s(), node_id=2)
        p = LiteralPattern(literal=lit, span=self._s(), node_id=1)
        assert p.literal is lit

    def test_var_pattern(self) -> None:
        p = VarPattern(name="x", span=self._s(), node_id=1)
        assert p.name == "x"

    def test_constructor_pattern_no_fields(self) -> None:
        p = ConstructorPattern(qualifier=None, name="None_", fields=(), span=self._s(), node_id=1)
        assert p.name == "None_"
        assert p.fields == ()

    def test_constructor_pattern_with_fields(self) -> None:
        pf = PatternField(
            name="val",
            pattern=VarPattern(name="v", span=self._s(), node_id=3),
            span=self._s(),
            node_id=2,
        )
        p = ConstructorPattern(
            qualifier=None,
            name="Some",
            fields=(pf,),
            span=self._s(),
            node_id=1,
        )
        assert isinstance(p.fields, tuple)
        assert p.fields[0].name == "val"

    def test_pattern_equality_ignores_span_node_id(self) -> None:
        a = WildcardPattern(span=span(1, 0, 1, 1), node_id=1)
        b = WildcardPattern(span=span(5, 0, 5, 1), node_id=50)
        assert a == b


# ---------------------------------------------------------------------------
# Program node
# ---------------------------------------------------------------------------

class TestProgram:
    def _s(self) -> SourceSpan:
        return span()

    def test_empty_program(self) -> None:
        prog = Program(body=(), span=self._s(), node_id=0)
        assert prog.body == ()

    def test_program_with_statements(self) -> None:
        st = PassStmt(span=self._s(), node_id=2)
        prog = Program(body=(st,), span=self._s(), node_id=1)
        assert isinstance(prog.body, tuple)
        assert len(prog.body) == 1

    def test_program_equality_ignores_span_node_id(self) -> None:
        st = PassStmt(span=span(1, 0, 1, 4), node_id=5)
        a = Program(body=(st,), span=span(1, 0, 1, 4), node_id=1)
        b = Program(body=(st,), span=span(9, 0, 9, 4), node_id=99)
        assert a == b


# ---------------------------------------------------------------------------
# Visitor / walk
# ---------------------------------------------------------------------------

class TestVisitorWalk:
    """Verify that walk() visits every node kind in a tree."""

    def _s(self) -> SourceSpan:
        return span()

    def _build_tree(self) -> Program:
        """Build a Program that exercises every node kind.

        Every type, expression, statement, declaration, and pattern node that
        exists in the AST must appear somewhere in the returned tree so that
        the walk-coverage tests can confirm they are all reachable.
        """
        s = self._s()

        # --- Type nodes (all 8 must appear as type_expr in some declaration) ---
        text_t = TextT(span=s, node_id=100)
        int_t = IntT(span=s, node_id=101)
        bool_t = BoolT(span=s, node_id=102)
        json_t = JsonT(span=s, node_id=103)
        decimal_t = DecimalT(span=s, node_id=104)
        name_t = NameT(name="MyT", span=s, node_id=105)
        list_t = ListT(elem=text_t, span=s, node_id=106)
        dict_t = DictT(value=int_t, span=s, node_id=107)

        # Declarations — each field uses a different type so all types appear.
        field_int = FieldDef(name="x", type_expr=int_t, span=s, node_id=200)
        field_bool = FieldDef(name="flag", type_expr=bool_t, span=s, node_id=201)
        field_json = FieldDef(name="data", type_expr=json_t, span=s, node_id=202)
        field_decimal = FieldDef(name="price", type_expr=decimal_t, span=s, node_id=203)
        field_name = FieldDef(name="ref", type_expr=name_t, span=s, node_id=204)
        field_dict = FieldDef(name="meta", type_expr=dict_t, span=s, node_id=205)
        record_def = RecordDef(
            name="Point",
            fields=(field_int, field_bool, field_json, field_decimal, field_name, field_dict),
            span=s,
            node_id=210,
        )

        variant_field = FieldDef(name="val", type_expr=text_t, span=s, node_id=211)
        variant_def = VariantDef(name="Some", fields=(variant_field,), span=s, node_id=212)
        enum_def = EnumDef(name="Opt", variants=(variant_def,), span=s, node_id=213)

        type_alias = TypeAlias(name="Names", type_expr=list_t, span=s, node_id=214)
        input_decl = InputDecl(name="spec", annotation=text_t, span=s, node_id=215)

        # --- Literals ---
        int_lit = IntLit(value=1, span=s, node_id=300)
        dec_lit = DecimalLit(value=decimal.Decimal("1.5"), span=s, node_id=301)
        bool_lit = BoolLit(value=True, span=s, node_id=302)
        null_lit = NullLit(span=s, node_id=303)
        str_lit = StringLit(value="hello", span=s, node_id=304)

        dict_entry = DictEntry(key=str_lit, value=int_lit, span=s, node_id=305)
        dict_lit = DictLit(entries=(dict_entry,), span=s, node_id=306)
        list_lit = ListLit(elements=(int_lit,), span=s, node_id=307)

        # --- Template ---
        text_seg = TextSegment(text="hi ", span=s, node_id=308)
        var_ref_tmpl = VarRef(name="name", span=s, node_id=309)
        interp_seg = InterpSegment(expr=var_ref_tmpl, span=s, node_id=310)
        template = Template(segments=(text_seg, interp_seg), span=s, node_id=311)

        # --- Expressions ---
        var_ref = VarRef(name="x", span=s, node_id=400)
        field_access = FieldAccess(obj=var_ref, field="y", span=s, node_id=401)
        named_arg = NamedArg(name="a", value=int_lit, span=s, node_id=402)
        constructor = Constructor(
            qualifier=None, name="Point", args=(named_arg,), span=s, node_id=403
        )

        abort_policy = AbortPolicy(span=s, node_id=405)
        retry_policy = RetryPolicy(extra=1, span=s, node_id=406)
        call_opts_full = CallOptions(
            format="json",
            strict_json=True,
            parse_policy=retry_policy,
            span=s,
            node_id=407,
        )
        agent_call = AgentCall(
            agent="ask", options=call_opts_full, template=template, span=s, node_id=408
        )

        binary_op = BinaryOp(op=BinOp.ADD, left=int_lit, right=int_lit, span=s, node_id=409)
        unary_not = UnaryNot(operand=bool_lit, span=s, node_id=410)
        unary_neg = UnaryNeg(operand=int_lit, span=s, node_id=411)
        is_test = IsTest(
            expr=var_ref, qualifier=None, variant="Some", negated=False, span=s, node_id=412
        )

        # --- Patterns ---
        wildcard_pat = WildcardPattern(span=s, node_id=500)
        lit_pat = LiteralPattern(literal=int_lit, span=s, node_id=501)
        var_pat = VarPattern(name="v", span=s, node_id=502)
        pat_field = PatternField(name="val", pattern=var_pat, span=s, node_id=503)
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Some", fields=(pat_field,), span=s, node_id=504
        )

        # CaseExpr with wildcard
        case_expr_branch = CaseExprBranch(pattern=wildcard_pat, body=null_lit, span=s, node_id=505)
        case_expr = CaseExpr(subject=var_ref, branches=(case_expr_branch,), span=s, node_id=506)

        # IfExpr with a cond branch and else branch
        if_expr_cond_branch = IfExprBranch(cond=bool_lit, body=int_lit, span=s, node_id=507)
        if_expr_else_branch = IfExprBranch(cond=ELSE, body=null_lit, span=s, node_id=508)
        if_expr = IfExpr(branches=(if_expr_cond_branch, if_expr_else_branch), span=s, node_id=509)

        # CallOptions with no policy (abort_policy embedded separately via ExprStmt)
        call_opts_none = CallOptions(
            format=None, strict_json=None, parse_policy=None, span=s, node_id=510
        )
        agent_call_abort = AgentCall(
            agent="exec",
            options=CallOptions(
                format=None, strict_json=None, parse_policy=abort_policy, span=s, node_id=511
            ),
            template=Template(segments=(), span=s, node_id=512),
            span=s,
            node_id=513,
        )

        # --- Statements ---
        let_decl = LetDecl(name="a", type_ann=None, value=int_lit, span=s, node_id=600)
        var_decl = VarDecl(name="b", type_ann=None, value=str_lit, span=s, node_id=601)
        set_stmt = SetStmt(target="b", value=int_lit, span=s, node_id=602)
        pass_stmt = PassStmt(span=s, node_id=603)
        print_stmt = PrintStmt(value=str_lit, span=s, node_id=604)
        expr_stmt_agent = ExprStmt(expr=agent_call, span=s, node_id=605)

        # More ExprStmts to get remaining expr kinds into the tree
        expr_stmt_fa = ExprStmt(expr=field_access, span=s, node_id=620)
        expr_stmt_bin = ExprStmt(expr=binary_op, span=s, node_id=621)
        expr_stmt_not = ExprStmt(expr=unary_not, span=s, node_id=622)
        expr_stmt_neg = ExprStmt(expr=unary_neg, span=s, node_id=623)
        expr_stmt_is = ExprStmt(expr=is_test, span=s, node_id=624)
        expr_stmt_ce = ExprStmt(expr=case_expr, span=s, node_id=625)
        expr_stmt_dl = ExprStmt(expr=dict_lit, span=s, node_id=626)
        expr_stmt_ll = ExprStmt(expr=list_lit, span=s, node_id=627)
        expr_stmt_dec = ExprStmt(expr=dec_lit, span=s, node_id=628)
        expr_stmt_abort = ExprStmt(expr=agent_call_abort, span=s, node_id=629)

        cond = BoolLit(value=True, span=s, node_id=606)
        do_until = DoUntil(limit=5, body=(pass_stmt,), condition=cond, span=s, node_id=607)

        if_branch = IfBranch(cond=bool_lit, body=(pass_stmt,), span=s, node_id=608)
        else_branch = IfBranch(cond=ELSE, body=(pass_stmt,), span=s, node_id=609)
        if_stmt = IfStmt(branches=(if_branch, else_branch), span=s, node_id=610)

        # CaseStmt with lit_pat to exercise LiteralPattern
        case_stmt_branch_lit = CaseStmtBranch(
            pattern=lit_pat, body=(pass_stmt,), span=s, node_id=630
        )
        case_stmt_branch_ctor = CaseStmtBranch(
            pattern=ctor_pat, body=(pass_stmt,), span=s, node_id=611
        )
        case_stmt = CaseStmt(
            subject=var_ref,
            branches=(case_stmt_branch_lit, case_stmt_branch_ctor),
            span=s,
            node_id=612,
        )

        catch_clause = CatchClause(
            exc_type="MyError", binding="e", body=(pass_stmt,), span=s, node_id=613
        )
        try_catch = TryCatch(body=(pass_stmt,), handlers=(catch_clause,), span=s, node_id=614)

        raise_stmt = Raise(exc=constructor, span=s, node_id=615)

        # LetDecl/VarDecl with type annotations to exercise type walk through stmts
        let_with_type = LetDecl(name="c", type_ann=bool_t, value=bool_lit, span=s, node_id=640)
        var_with_type = VarDecl(name="d", type_ann=json_t, value=null_lit, span=s, node_id=641)

        # AgentCall with parse_policy=None to exercise the None branch in walk(CallOptions)
        agent_call_no_policy = AgentCall(
            agent="ask",
            options=call_opts_none,
            template=Template(segments=(), span=s, node_id=650),
            span=s,
            node_id=651,
        )
        expr_stmt_no_policy = ExprStmt(expr=agent_call_no_policy, span=s, node_id=652)
        expr_stmt_ie = ExprStmt(expr=if_expr, span=s, node_id=653)

        body = (
            record_def, enum_def, type_alias, input_decl,
            let_decl, let_with_type, var_decl, var_with_type,
            set_stmt, pass_stmt, print_stmt,
            expr_stmt_agent, expr_stmt_fa, expr_stmt_bin, expr_stmt_not,
            expr_stmt_neg, expr_stmt_is, expr_stmt_ce, expr_stmt_dl,
            expr_stmt_ll, expr_stmt_dec, expr_stmt_abort, expr_stmt_no_policy,
            expr_stmt_ie,
            do_until, if_stmt, case_stmt, try_catch, raise_stmt,
        )

        return Program(body=body, span=s, node_id=0)

    def test_walk_visits_program(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Program) for n in visited)

    def test_walk_visits_all_stmt_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        stmt_kinds = {
            LetDecl, VarDecl, SetStmt, PassStmt, PrintStmt, ExprStmt,
            DoUntil, IfStmt, IfBranch, CaseStmt, CaseStmtBranch,
            TryCatch, CatchClause, Raise,
        }
        for kind in stmt_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_decl_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        decl_kinds = {RecordDef, EnumDef, TypeAlias, InputDecl, FieldDef, VariantDef}
        for kind in decl_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_agent_decl(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = AgentDecl(name="reviewer", runner=None, span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        # AgentDecl is a leaf — only itself is visited.
        assert visited == [node]

    def test_walk_input_decl_without_annotation(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = span()
        node = InputDecl(name="spec", annotation=None, span=s, node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        # Only the InputDecl itself is visited; the missing annotation adds no child.
        assert visited == [node]

    def test_walk_visits_all_expr_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        expr_kinds = {
            VarRef, FieldAccess, AgentCall, Constructor, NamedArg,
            BinaryOp, UnaryNot, UnaryNeg, IsTest, CaseExpr, CaseExprBranch,
            IfExpr, IfExprBranch,
            IntLit, DecimalLit, BoolLit, NullLit, StringLit, ListLit, DictLit,
            DictEntry, Template, TextSegment, InterpSegment,
        }
        for kind in expr_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_type_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        type_kinds = {TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT}
        for kind in type_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_pattern_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        pattern_kinds = {
            WildcardPattern, LiteralPattern, VarPattern, ConstructorPattern, PatternField
        }
        for kind in pattern_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_call_option_nodes(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        assert CallOptions in kinds
        assert RetryPolicy in kinds

    def test_visitor_subclass(self) -> None:
        """Subclassing Visitor and overriding visit_PassStmt should be called."""
        from agm.agl.syntax.visitor import Visitor, walk

        s = self._s()
        prog = Program(
            body=(PassStmt(span=s, node_id=2),),
            span=s,
            node_id=1,
        )

        class CountPass(Visitor):
            def __init__(self) -> None:
                self.count = 0

            def visit_PassStmt(self, node: PassStmt) -> None:
                self.count += 1

        counter = CountPass()
        walk(prog, counter.dispatch)
        assert counter.count == 1

    def test_visitor_dispatch_unknown_type_raises(self) -> None:
        """Visitor.dispatch on an unknown type should raise loudly."""
        from agm.agl.syntax.visitor import Visitor

        class MyVisitor(Visitor):
            pass

        v = MyVisitor()

        class NotANode:
            pass

        with pytest.raises(TypeError):
            v.dispatch(NotANode())

    def test_walk_unknown_type_raises(self) -> None:
        """walk() on an unknown type should raise TypeError."""
        from agm.agl.syntax.visitor import walk

        class NotANode:
            pass

        with pytest.raises(TypeError):
            walk(NotANode(), lambda n: None)


# ---------------------------------------------------------------------------
# Union alias sanity
# ---------------------------------------------------------------------------

class TestUnionAliases:
    """Verify that the union aliases are populated (not just None/object)."""

    def test_pass_stmt_is_stmt(self) -> None:
        import typing

        args = typing.get_args(Stmt)
        assert PassStmt in args

    def test_var_ref_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert VarRef in args

    def test_wildcard_pattern_is_pattern(self) -> None:
        import typing

        args = typing.get_args(Pattern)
        assert WildcardPattern in args

    def test_text_segment_is_template_segment(self) -> None:
        import typing

        args = typing.get_args(TemplateSegment)
        assert TextSegment in args
