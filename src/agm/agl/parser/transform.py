"""AstBuilder: Lark tree → agm.agl.syntax.* (the parser firewall).

This is the ONLY module in the codebase that imports both ``lark`` and
``agm.agl.syntax``.  Everything downstream of this module depends only on
the ``agm.agl.syntax`` dataclasses — never on Lark.

Design
------
- ``AstBuilder(lark.Transformer)`` with ``@v_args(meta=True)`` at class level
  so every rule method receives ``meta`` as its first positional argument.
- Monotonic ``node_id`` counter assigned per-node in document order.
- ``SourceSpan`` is derived from ``meta.line`` / ``meta.column`` /
  ``meta.end_line`` / ``meta.end_column`` / ``meta.start_pos`` /
  ``meta.end_pos`` — all provided by Lark when ``propagate_positions=True``.
- Both ``span`` and ``node_id`` are keyword-only ``compare=False`` fields in
  every AST dataclass, so equality tests ignore them.

Span convention for tokens used as leaves
------------------------------------------
When a rule has a single Token child and no meta (e.g. ``var_ref``), the
token's own position fields are used directly.  Rule-level meta is preferred
because it covers the full span of multi-token productions.
"""

from __future__ import annotations

import decimal
from itertools import count
from typing import cast

from lark import Transformer, v_args
from lark.lexer import Token
from lark.tree import Meta

import agm.agl.syntax as syntax
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.syntax.nodes import ELSE, PragmaValue
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    AgentT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
    TypeExpr,
    UnitT,
)

# Types used internally
_NamedArgList = list[syntax.NamedArg]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Args = list[object]  # Rule children after transformation (tokens + AST nodes)

_ALL_TYPE_EXPRS = (TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT, UnitT, AgentT, FuncT)


def _span_from_meta(meta: Meta) -> SourceSpan:
    """Build a SourceSpan from Lark tree Meta (propagate_positions=True)."""
    return SourceSpan(
        start_line=meta.line,
        start_col=meta.column,
        end_line=meta.end_line,
        end_col=meta.end_column,
        start_offset=meta.start_pos,
        end_offset=meta.end_pos,
    )


def _span_from_token(tok: Token) -> SourceSpan:
    """Build a SourceSpan from a Lark Token's position fields."""
    line = tok.line if tok.line is not None else 1
    col = tok.column if tok.column is not None else 1
    pos = tok.start_pos if tok.start_pos is not None else 0
    end_line = tok.end_line if tok.end_line is not None else line
    end_col = tok.end_column if tok.end_column is not None else col + len(str(tok))
    end_pos = tok.end_pos if tok.end_pos is not None else pos + len(str(tok))
    return SourceSpan(
        start_line=line,
        start_col=col,
        end_line=end_line,
        end_col=end_col,
        start_offset=pos,
        end_offset=end_pos,
    )


# ---------------------------------------------------------------------------
# AstBuilder
# ---------------------------------------------------------------------------


@v_args(meta=True)
class AstBuilder(Transformer):
    """Transforms a Lark parse tree into ``agm.agl.syntax`` dataclasses.

    The ``node_id`` counter is monotonically increasing and starts at
    ``start_id`` (default ``0``).  Each ``parse_program`` call creates a fresh
    builder, so node IDs within a single program are deterministic (assigned
    in tree-walk order — root first, depth-first left-to-right).  Incremental
    sessions seed ``start_id`` from a prior parse's ``next_node_id`` so ids
    stay globally unique across entries.
    """

    def __init__(self, *, start_id: int = 0) -> None:
        super().__init__()
        self._counter = count(start_id)
        # The next id the counter will hand out.  Tracked explicitly so callers
        # can read the first id NOT consumed after a transform (the seed for a
        # subsequent incremental parse) without having to assume the root node
        # holds the maximum id.  Seeded to ``start_id`` before any node is built.
        self._next_unused: int = start_id
        # node_ids of Constructor nodes that already had a constructor_payload
        # applied via ctor_applied.  Used to reject a second payload
        # (``Issue(a: 1)(b: 2)``) regardless of whether the first payload was
        # empty (``Empty()``).
        self._payload_applied: set[int] = set()

    def _next_id(self) -> int:
        nid = next(self._counter)
        self._next_unused = nid + 1
        return nid

    @property
    def next_node_id(self) -> int:
        """The first ``node_id`` NOT yet consumed by this builder.

        After ``transform`` this is the seed (``start_id``) for the next
        incremental parse so that node ids stay globally unique across entries.
        Equal to ``start_id`` when no node has been built.
        """
        return self._next_unused

    # ------------------------------------------------------------------
    # Program root
    # ------------------------------------------------------------------

    def start(self, meta: Meta, args: _Args) -> syntax.Program:
        (block,) = args
        assert isinstance(block, syntax.Block)
        span = _span_from_meta(meta)
        return syntax.Program(body=block, span=span, node_id=self._next_id())

    def block(self, meta: Meta, args: _Args) -> syntax.Block:
        """block: item (_sep item)* _sep?

        _NEWLINE tokens are filtered by leading underscore convention.
        SEMICOLON tokens are %declare'd and appear in tree — drop them.
        items are Declaration | Binder | Expr (all are AST nodes, not Tokens).
        """
        items = tuple(
            cast(syntax.Item, a)
            for a in args
            if a is not None and not isinstance(a, Token)
        )
        return syntax.Block(
            items=items,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def param_decl(self, meta: Meta, args: _Args) -> syntax.ParamDecl:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        ann, default = _extract_ann_and_optional_expr(args[1:])
        span = _span_from_meta(meta)
        return syntax.ParamDecl(
            name=str(name_tok),
            annotation=ann,
            default=default,
            span=span,
            node_id=self._next_id(),
        )

    def program_decl(self, meta: Meta, args: _Args) -> syntax.ProgramDecl:
        name_tok = next(a for a in args if isinstance(a, Token) and a.type == "VAR_NAME")
        span = _span_from_meta(meta)
        return syntax.ProgramDecl(
            name=str(name_tok),
            span=span,
            node_id=self._next_id(),
        )

    def agent_decl(self, meta: Meta, args: _Args) -> syntax.AgentDecl:
        # Grammar: AGENT VAR_NAME (EQ template)?
        name_tok = next(
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        )
        runner_node = next(
            (a for a in args if isinstance(a, (syntax.StringLit, syntax.Template))),
            None,
        )
        runner: str | None = (
            None
            if runner_node is None
            else _require_literal_string(
                runner_node,
                "agent runner string must be a literal string with no "
                "interpolation.",
            ).value
        )
        span = _span_from_meta(meta)
        return syntax.AgentDecl(
            name=str(name_tok),
            runner=runner,
            span=span,
            node_id=self._next_id(),
        )

    def config_pragma(self, meta: Meta, args: _Args) -> syntax.ConfigPragma:
        """config_pragma: "config" VAR_NAME EQ pragma_value"""
        key_tok = next(a for a in args if isinstance(a, Token) and a.type == "VAR_NAME")
        raw_value = next(
            a for a in args
            if a is not None and not isinstance(a, Token)
        )
        span = _span_from_meta(meta)
        return syntax.ConfigPragma(
            key=str(key_tok),
            value=cast(PragmaValue, raw_value),
            span=span,
            node_id=self._next_id(),
        )

    def pragma_true(self, meta: Meta, args: _Args) -> bool:
        return True

    def pragma_false(self, meta: Meta, args: _Args) -> bool:
        return False

    def pragma_int(self, meta: Meta, args: _Args) -> int:
        tok = args[0]
        assert isinstance(tok, Token)
        return int(str(tok))

    def pragma_decimal(self, meta: Meta, args: _Args) -> decimal.Decimal:
        tok = args[0]
        assert isinstance(tok, Token)
        return decimal.Decimal(str(tok))

    def pragma_str(self, meta: Meta, args: _Args) -> str:
        lit = _require_literal_string(
            args[0],
            "config pragma value must be a literal string with no interpolation.",
        )
        return lit.value

    # ------------------------------------------------------------------
    # record_def / field_def
    # ------------------------------------------------------------------

    def record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        # Grammar: "record" TYPE_NAME _NEWLINE _INDENT field_def+ _DEDENT
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        fields = tuple(a for a in args[1:] if isinstance(a, syntax.FieldDef))
        return syntax.RecordDef(
            name=str(name_tok),
            fields=fields,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def field_def(self, meta: Meta, args: _Args) -> syntax.FieldDef:
        # Grammar: field_name COLON type_expr
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        type_expr = _find_type_expr(args[1:])
        return syntax.FieldDef(
            name=str(name_tok),
            type_expr=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # enum_def / variant_def / variant_payload / field_list / field_inline
    # ------------------------------------------------------------------

    def enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        variants = tuple(a for a in args[1:] if isinstance(a, syntax.VariantDef))
        return syntax.EnumDef(
            name=str(name_tok),
            variants=variants,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def variant_def(self, meta: Meta, args: _Args) -> syntax.VariantDef:
        # Grammar: PIPE TYPE_NAME variant_payload?
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        assert name_tok is not None, "variant_def: no TYPE_NAME token"
        fields: tuple[syntax.FieldDef, ...] = ()
        for a in args:
            if isinstance(a, tuple):
                fields = cast(tuple[syntax.FieldDef, ...], a)
                break
        return syntax.VariantDef(
            name=str(name_tok),
            fields=fields,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def variant_payload(self, meta: Meta, args: _Args) -> tuple[syntax.FieldDef, ...]:
        # Grammar: LPAR field_list? RPAR
        for a in args:
            if isinstance(a, tuple):
                return cast(tuple[syntax.FieldDef, ...], a)
        return ()

    def field_list(self, meta: Meta, args: _Args) -> tuple[syntax.FieldDef, ...]:
        # Grammar: field_inline (COMMA field_inline)* COMMA?
        return tuple(a for a in args if isinstance(a, syntax.FieldDef))

    # Grammar: VAR_NAME COLON type_expr — identical shape to ``field_def``.
    field_inline = field_def

    # ------------------------------------------------------------------
    # type_alias
    # ------------------------------------------------------------------

    def type_alias(self, meta: Meta, args: _Args) -> syntax.TypeAlias:
        # Grammar: "type" TYPE_NAME EQ type_expr
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        assert name_tok is not None, "type_alias: no TYPE_NAME token"
        type_expr = _find_type_expr(args)
        return syntax.TypeAlias(
            name=str(name_tok),
            type_expr=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # func_def / param_list / param_def / func_body
    # ------------------------------------------------------------------

    def func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """func_def: "def" VAR_NAME LPAR param_list? RPAR THIN_ARROW type_expr EQ func_body"""
        name_tok = next(a for a in args if isinstance(a, Token) and a.type == "VAR_NAME")
        params, return_type, body = self._split_params_type_body(args)
        assert return_type is not None, "func_def: no return type"
        assert body is not None, "func_def: no body"
        return syntax.FuncDef(
            name=str(name_tok),
            params=params,
            return_type=return_type,
            body=body,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def param_list(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        """param_list: param_def (COMMA param_def)* COMMA?"""
        return tuple(a for a in args if isinstance(a, syntax.Param))

    def param_def(self, meta: Meta, args: _Args) -> syntax.Param:
        """param_def: VAR_NAME COLON type_expr (EQ or_expr)?"""
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        type_expr = _find_type_expr(args[1:])
        # Default value: the or_expr after EQ, if present.
        # After the type_expr, look for any Expr (skip Tokens and TypeExprs).
        default: syntax.Expr | None = None
        seen_type = False
        for a in args[1:]:
            if isinstance(a, _ALL_TYPE_EXPRS):
                seen_type = True
                continue
            if seen_type and a is not None and not isinstance(a, Token):
                default = cast(syntax.Expr, a)
                break
        return syntax.Param(
            name=str(name_tok),
            type_expr=type_expr,
            default=default,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def func_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """func_body: suite_expr | expr — pass through the inner expr."""
        # args[0] is a Block (from suite_expr) or any Expr (from expr).
        expr = _find_non_token(args)
        return cast(syntax.Expr, expr)

    # ------------------------------------------------------------------
    # let_decl / var_decl / set_stmt
    # ------------------------------------------------------------------

    def let_decl(self, meta: Meta, args: _Args) -> syntax.LetDecl:
        # Grammar: "let" VAR_NAME type_ann? EQ expr
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        ann, value = _extract_ann_and_value(args[1:])
        span = _span_from_meta(meta)
        return syntax.LetDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def var_decl(self, meta: Meta, args: _Args) -> syntax.VarDecl:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        ann, value = _extract_ann_and_value(args[1:])
        span = _span_from_meta(meta)
        return syntax.VarDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def set_target(self, meta: Meta, args: _Args) -> syntax.SetTarget:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        indexes = [cast(syntax.Expr, a) for a in args[1:] if _is_expr_node(a)]
        if not indexes:
            return syntax.NameTarget(
                name=str(name_tok),
                span=_span_from_token(name_tok),
                node_id=self._next_id(),
            )

        obj: syntax.Expr = syntax.VarRef(
            name=str(name_tok),
            span=_span_from_token(name_tok),
            node_id=self._next_id(),
        )
        for index in indexes[:-1]:
            obj = syntax.IndexAccess(
                obj=obj,
                index=index,
                span=_span_from_meta(meta),
                node_id=self._next_id(),
            )
        return syntax.IndexTarget(
            obj=obj,
            index=indexes[-1],
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def set_stmt(self, meta: Meta, args: _Args) -> syntax.SetStmt:
        # Grammar: "set" set_target EQ expr
        target = next(a for a in args if _is_set_target(a))
        value = cast(syntax.Expr, next(a for a in args if _is_expr_node(a)))
        span = _span_from_meta(meta)
        return syntax.SetStmt(
            target=cast(syntax.SetTarget, target),
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # type_ann
    # ------------------------------------------------------------------

    def type_ann(self, meta: Meta, args: _Args) -> TypeExpr:
        # Grammar: type_ann: COLON type_expr
        return _find_type_expr(args)

    # ------------------------------------------------------------------
    # type_expr dispatch (grammar rule names)
    # ------------------------------------------------------------------

    def prim_type_or_name(self, meta: Meta, args: _Args) -> TypeExpr:
        """VAR_NAME used in type position — map to primitive or NameT."""
        tok = args[0]
        assert isinstance(tok, Token)
        name = str(tok)
        span = _span_from_meta(meta)
        nid = self._next_id()
        if name == "text":
            return TextT(span=span, node_id=nid)
        if name == "json":
            return JsonT(span=span, node_id=nid)
        if name == "bool":
            return BoolT(span=span, node_id=nid)
        if name == "int":
            return IntT(span=span, node_id=nid)
        if name == "decimal":
            return DecimalT(span=span, node_id=nid)
        if name == "unit":
            return UnitT(span=span, node_id=nid)
        # Anything else is a named type reference.
        return NameT(name=name, span=span, node_id=nid)

    def named_type(self, meta: Meta, args: _Args) -> NameT:
        """TYPE_NAME used in type position."""
        tok = args[0]
        assert isinstance(tok, Token)
        return NameT(name=str(tok), span=_span_from_meta(meta), node_id=self._next_id())

    def agent_type(self, meta: Meta, args: _Args) -> AgentT:
        """AGENT terminal in type position → AgentT."""
        return AgentT(span=_span_from_meta(meta), node_id=self._next_id())

    def generic_type_1(self, meta: Meta, args: _Args) -> TypeExpr:
        """VAR_NAME LSQB type_expr RSQB — handles list[T]."""
        head_tok = args[0]
        assert isinstance(head_tok, Token)
        head = str(head_tok)
        inner: TypeExpr = _find_type_expr(args[1:])
        span = _span_from_meta(meta)
        nid = self._next_id()
        if head == "list":
            return ListT(elem=inner, span=span, node_id=nid)
        raise syntax_error_from_meta(meta, f"Unknown generic type: {head!r}")

    def dict_type(self, meta: Meta, args: _Args) -> DictT:
        """VAR_NAME LSQB VAR_NAME COMMA type_expr RSQB — dict[text, V]."""
        var_name_toks = [
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        ]
        assert len(var_name_toks) >= 2, "dict_type: missing key token"
        head_tok = var_name_toks[0]
        if str(head_tok) != "dict":
            raise syntax_error_from_meta(meta, f"Unknown generic type: {str(head_tok)!r}")
        key_tok = var_name_toks[1]
        if str(key_tok) != "text":
            raise AglSyntaxError(
                f"dict keys are always text in v1, got {str(key_tok)!r}.",
                span=_span_from_token(key_tok),
            )
        value: TypeExpr = _find_type_expr(args[1:])
        return DictT(value=value, span=_span_from_meta(meta), node_id=self._next_id())

    def func_type(self, meta: Meta, args: _Args) -> FuncT:
        """LPAR type_list? RPAR THIN_ARROW type_expr — function type (A, B) -> C."""
        # All TypeExpr nodes in args; the last one is the result type.
        # The type_list (if present) produces a tuple of TypeExprs as a tuple.
        # With maybe_placeholders=True, absent type_list is None.
        param_types: tuple[TypeExpr, ...] = ()
        result_type: TypeExpr | None = None
        for a in args:
            if isinstance(a, tuple) and all(isinstance(x, _ALL_TYPE_EXPRS) for x in a):
                # type_list result
                param_types = cast(tuple[TypeExpr, ...], a)
            elif isinstance(a, _ALL_TYPE_EXPRS):
                # The result type (last TypeExpr child after THIN_ARROW)
                result_type = a
            # None (placeholder for absent type_list) and Tokens are skipped
        assert result_type is not None, "func_type: no result type"
        return FuncT(
            params=param_types,
            result=result_type,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def type_list(self, meta: Meta, args: _Args) -> tuple[TypeExpr, ...]:
        """type_list: type_expr (COMMA type_expr)* COMMA?"""
        return tuple(a for a in args if isinstance(a, _ALL_TYPE_EXPRS))

    # ------------------------------------------------------------------
    # unit_lit / paren_expr
    # ------------------------------------------------------------------

    def unit_lit(self, meta: Meta, args: _Args) -> syntax.UnitLit:
        return syntax.UnitLit(span=_span_from_meta(meta), node_id=self._next_id())

    def paren_expr(self, meta: Meta, args: _Args) -> syntax.Expr:
        # args: LPAR, expr, RPAR — find the expr (non-Token)
        for a in args:
            if not isinstance(a, Token) and a is not None:
                return cast(syntax.Expr, a)
        raise AssertionError("paren_expr: no inner expression found")  # pragma: no cover

    # ------------------------------------------------------------------
    # Lambda expression
    # ------------------------------------------------------------------

    def _split_params_type_body(
        self, args: _Args
    ) -> tuple[tuple[syntax.Param, ...], TypeExpr | None, syntax.Expr | None]:
        """Classify a func/lambda arg list into ``(params, return_type, body)``.

        Shared by ``func_def`` (return type required) and ``lambda_expr`` (return
        type optional); callers assert on the parts they require.
        """
        params: tuple[syntax.Param, ...] = ()
        return_type: TypeExpr | None = None
        body: syntax.Expr | None = None
        for a in args:
            if isinstance(a, tuple) and all(isinstance(x, syntax.Param) for x in a):
                params = cast(tuple[syntax.Param, ...], a)
            elif isinstance(a, _ALL_TYPE_EXPRS):
                return_type = a
            elif a is not None and not isinstance(a, Token) and not isinstance(a, tuple):
                body = cast(syntax.Expr, a)
        return params, return_type, body

    def lambda_expr(self, meta: Meta, args: _Args) -> syntax.Lambda:
        """lambda_expr: "fn" LPAR param_list? RPAR (THIN_ARROW type_expr)? ARROW expr"""
        params, return_type, body = self._split_params_type_body(args)
        assert body is not None, "lambda_expr: no body"
        return syntax.Lambda(
            params=params,
            return_type=return_type,
            body=body,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def lit_int(self, meta: Meta, args: _Args) -> syntax.IntLit:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.IntLit(
            value=int(str(tok)),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_decimal(self, meta: Meta, args: _Args) -> syntax.DecimalLit:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.DecimalLit(
            value=decimal.Decimal(str(tok)),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_true(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(
            value=True, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def lit_false(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(
            value=False, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def lit_null(self, meta: Meta, args: _Args) -> syntax.NullLit:
        return syntax.NullLit(span=_span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # var_ref / constructor
    # ------------------------------------------------------------------

    def var_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.VarRef(
            name=str(tok), span=_span_from_meta(meta), node_id=self._next_id()
        )

    def ctor_unqualified(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """constructor: TYPE_NAME → bare unqualified constructor (no args)."""
        name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None
        )
        assert name_tok is not None, "ctor_unqualified: no TYPE_NAME token"
        return syntax.Constructor(
            qualifier=None,
            name=str(name_tok),
            args=(),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Postfix: call / field_access / type_access / ctor_applied
    # ------------------------------------------------------------------

    def call(self, meta: Meta, args: _Args) -> syntax.Call | syntax.Constructor:
        """postfix LPAR arg_list? RPAR → Call node, or Constructor when callee is TYPE_NAME.

        Constructor application: when the callee is a bare Constructor node
        (e.g. ``Issue``, ``Review.Pass``), we route the named args into the
        Constructor rather than producing a Call.  Constructors only take named
        args in v1; positional args on a constructor are rejected here.

        When the callee is a Constructor that already had args applied (double
        payload ``Issue(a:1)(b:2)``), we reject it.

        For all other callees, we produce a Call node.
        """
        # args[0] is the callee (postfix result, any Expr)
        callee = cast(syntax.Expr, args[0])
        # Remaining args: optional arg_list result, Tokens (LPAR/RPAR)
        pos_args: list[syntax.Expr] = []
        named_args: list[syntax.NamedArg] = []
        for a in args[1:]:
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], list):
                # arg_list returned a (pos_args, named_args) pair
                pa, na = cast(tuple[list[syntax.Expr], list[syntax.NamedArg]], a)
                pos_args = pa
                named_args = na
            # Tokens (LPAR, RPAR) and None are skipped

        span = _span_from_meta(meta)

        # Handle constructor application: Issue(title: x, severity: 1)
        if isinstance(callee, syntax.Constructor):
            if callee.node_id in self._payload_applied:
                raise AglSyntaxError(
                    "constructor takes a single argument list.",
                    span=self._span_after(callee.span, meta),
                )
            if pos_args:
                raise AglSyntaxError(
                    "constructor arguments must be named (e.g. Issue(title: x)).",
                    span=span,
                )
            new_id = self._next_id()
            self._payload_applied.add(new_id)
            return syntax.Constructor(
                qualifier=callee.qualifier,
                name=callee.name,
                args=tuple(named_args),
                span=span,
                node_id=new_id,
            )

        return syntax.Call(
            callee=callee,
            args=tuple(pos_args),
            named_args=tuple(named_args),
            span=span,
            node_id=self._next_id(),
        )

    def field_access(self, meta: Meta, args: _Args) -> syntax.FieldAccess:
        """postfix DOT field_name — record field access."""
        obj_expr = cast(syntax.Expr, args[0])
        field_tok = _find_name_token(args)
        return syntax.FieldAccess(
            obj=obj_expr,
            field=str(field_tok),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def index_access(self, meta: Meta, args: _Args) -> syntax.IndexAccess:
        """postfix INDEX_LSQB expr RSQB — list/dict index access."""
        exprs = [a for a in args if _is_expr_node(a)]
        obj_expr, index_expr = exprs
        return syntax.IndexAccess(
            obj=cast(syntax.Expr, obj_expr),
            index=cast(syntax.Expr, index_expr),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def type_access(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """postfix DOT TYPE_NAME — qualified enum-variant constructor.

        Only ``TYPE_NAME . TYPE_NAME`` qualification is legal.  The LHS must
        be a bare, unqualified, no-arg ``Constructor``.
        """
        obj_expr = args[0]
        type_name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None
        )
        assert type_name_tok is not None, "type_access: no TYPE_NAME token"
        variant_name = str(type_name_tok)
        if (
            isinstance(obj_expr, syntax.Constructor)
            and obj_expr.qualifier is None
            and not obj_expr.args
            and obj_expr.node_id not in self._payload_applied
        ):
            return syntax.Constructor(
                qualifier=obj_expr.name,
                name=variant_name,
                args=(),
                span=_span_from_meta(meta),
                node_id=self._next_id(),
            )
        raise AglSyntaxError(
            f"'.{variant_name}' may only follow a type name "
            "(qualified constructor); chained or non-type qualification is "
            "not allowed.",
            span=_span_from_token(type_name_tok),
        )

    @staticmethod
    def _span_after(base: SourceSpan, meta: Meta) -> SourceSpan:
        """Span covering the payload that follows ``base`` within the rule meta."""
        return SourceSpan(
            start_line=base.end_line,
            start_col=base.end_col,
            end_line=meta.end_line,
            end_col=meta.end_column,
            start_offset=base.end_offset,
            end_offset=meta.end_pos,
        )

    # ------------------------------------------------------------------
    # Juxtaposition (single-arg sugar)
    # ------------------------------------------------------------------

    def juxt(self, meta: Meta, args: _Args) -> syntax.Call | syntax.Expr:
        """juxt: postfix juxt_arg -> juxt_call | postfix

        The postfix-only alternative wraps nothing — just returns the postfix expr.
        The juxt_call alternative builds a Call with one positional arg.
        This method is never called directly because both alternatives have
        explicit aliases; see juxt_call below and the transparent postfix passthrough.
        """
        # This method is called for the `postfix` alternative (no alias).
        # The `postfix juxt_arg` alternative uses `-> juxt_call` alias.
        (inner,) = [a for a in args if a is not None and not isinstance(a, Token)]
        return cast(syntax.Expr, inner)

    def juxt_call(self, meta: Meta, args: _Args) -> syntax.Call:
        """juxt: postfix juxt_arg -> juxt_call

        Single-arg sugar: `f x` desugars to `Call(callee=f, args=(x,), named_args=())`.
        """
        # args[0] is the callee (postfix result)
        # args[1] is the juxt_arg result (an Expr)
        callee = cast(syntax.Expr, args[0])
        arg_expr = cast(syntax.Expr, args[1])
        return syntax.Call(
            callee=callee,
            args=(arg_expr,),
            named_args=(),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def juxt_bare(self, meta: Meta, args: _Args) -> syntax.Expr:
        """juxt_arg: juxt_atom -> juxt_bare — pass through the atom."""
        (inner,) = [a for a in args if a is not None and not isinstance(a, Token)]
        return cast(syntax.Expr, inner)

    def juxt_field_access(self, meta: Meta, args: _Args) -> syntax.FieldAccess:
        """juxt_arg: juxt_atom (DOT field_name)+ -> juxt_field_access

        Builds a chain of FieldAccess nodes for `print res.stdout`.
        The first non-Token non-None arg is the base; the remaining name
        tokens (from DOT field_name repetitions) are the field chain.
        """
        # The base atom is the sole non-Token arg; the field chain is the
        # VAR_NAME/AGENT tokens from the DOT repetitions.
        field_names = [
            str(a) for a in args if isinstance(a, Token) and a.type in ("VAR_NAME", "AGENT")
        ]
        non_tokens = [a for a in args if not isinstance(a, Token)]
        assert non_tokens, "juxt_field_access: no base atom"
        assert field_names, "juxt_field_access: no field names"
        result: syntax.Expr = cast(syntax.Expr, non_tokens[0])
        for fname in field_names:
            result = syntax.FieldAccess(
                obj=result,
                field=fname,
                span=_span_from_meta(meta),
                node_id=self._next_id(),
            )
        return cast(syntax.FieldAccess, result)

    # ------------------------------------------------------------------
    # Typed call (callee::[Type](args))
    # ------------------------------------------------------------------

    def typed_call_atom(self, meta: Meta, args: _Args) -> syntax.Call:
        """typed_call_atom: VAR_NAME DCOLON LSQB type_expr RSQB LPAR arg_list? RPAR

        Builds a ``Call`` whose callee is a bare ``VarRef`` and whose
        ``type_arg`` carries the static type expression.  The resolver /
        checker / interpreter treat the callee name as a built-in (currently
        only ``ask-request``); any other name is a scope error.
        """
        name_tok = next(
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        )
        type_expr = _find_type_expr(args)
        assert type_expr is not None, "typed_call_atom: no type_expr found"

        pos_args: list[syntax.Expr] = []
        named_args: list[syntax.NamedArg] = []
        for a in args:
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], list):
                pa, na = cast(tuple[list[syntax.Expr], list[syntax.NamedArg]], a)
                pos_args = pa
                named_args = na

        callee = syntax.VarRef(
            name=str(name_tok),
            span=_span_from_token(name_tok),
            node_id=self._next_id(),
        )
        return syntax.Call(
            callee=callee,
            args=tuple(pos_args),
            named_args=tuple(named_args),
            type_arg=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Call arguments
    # ------------------------------------------------------------------

    def arg_list(
        self, meta: Meta, args: _Args
    ) -> tuple[list[syntax.Expr], list[syntax.NamedArg]]:
        """arg_list: arg (COMMA arg)* COMMA?

        Returns (pos_args, named_args) pair for the call builder.
        Duplicate named arg names are rejected with the span of the duplicate.
        """
        pos_args: list[syntax.Expr] = []
        named_args: list[syntax.NamedArg] = []
        seen_names: dict[str, SourceSpan] = {}
        for a in args:
            if isinstance(a, syntax.NamedArg):
                if a.name in seen_names:
                    raise AglSyntaxError(
                        f"duplicate argument {a.name!r}.",
                        span=a.span,
                    )
                seen_names[a.name] = a.span
                named_args.append(a)
            elif a is not None and not isinstance(a, Token):
                # pos_arg (transparent ?) — the expr itself
                pos_args.append(cast(syntax.Expr, a))
        return (pos_args, named_args)

    def pos_arg(self, meta: Meta, args: _Args) -> syntax.Expr:
        """pos_arg: expr — transparent wrapper; return the expr."""
        return _find_expr(args)

    def named_arg(self, meta: Meta, args: _Args) -> syntax.NamedArg:
        """named_arg: field_name COLON expr"""
        name_tok = _find_name_token(args)
        val_expr = _find_expr(args[1:])
        return syntax.NamedArg(
            name=str(name_tok),
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Binary operators
    # ------------------------------------------------------------------

    def _binary(self, meta: Meta, args: _Args, op: syntax.BinOp) -> syntax.BinaryOp:
        left = cast(syntax.Expr, args[0])
        right = cast(syntax.Expr, args[-1])
        return syntax.BinaryOp(
            op=op, left=left, right=right,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def bin_or(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.OR)

    def bin_and(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.AND)

    def bin_eq(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.EQ)

    def bin_neq(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.NEQ)

    def bin_lt(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.LT)

    def bin_le(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.LE)

    def bin_gt(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.GT)

    def bin_ge(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.GE)

    def bin_in(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.IN)

    def bin_add(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.ADD)

    def bin_sub(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.SUB)

    def bin_mul(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.MUL)

    def bin_div(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.DIV)

    # ------------------------------------------------------------------
    # Unary operators
    # ------------------------------------------------------------------

    def unary_not(self, meta: Meta, args: _Args) -> syntax.UnaryNot:
        operand = cast(syntax.Expr, args[0])
        return syntax.UnaryNot(
            operand=operand, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def unary_neg(self, meta: Meta, args: _Args) -> syntax.UnaryNeg:
        operand = cast(syntax.Expr, args[-1])
        return syntax.UnaryNeg(
            operand=operand, span=_span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # is / is not tests
    # ------------------------------------------------------------------

    def _make_is_test(
        self, meta: Meta, args: _Args, *, qualified: bool, negated: bool
    ) -> syntax.IsTest:
        left = cast(syntax.Expr, args[0])
        type_toks = [a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"]
        if qualified:
            assert len(type_toks) == 2
            qualifier: str | None = str(type_toks[0])
            variant = str(type_toks[1])
        else:
            qualifier = None
            variant = str(type_toks[0])
        return syntax.IsTest(
            expr=left, qualifier=qualifier, variant=variant, negated=negated,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def is_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, qualified=False, negated=False)

    def is_test_qualified(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, qualified=True, negated=False)

    def is_not_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, qualified=False, negated=True)

    def is_not_test_qualified(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, qualified=True, negated=True)

    # ------------------------------------------------------------------
    # Control flow: if_expr
    # ------------------------------------------------------------------

    def if_cond_branch(self, meta: Meta, args: _Args) -> syntax.IfBranch:
        """if_cond_branch: or_expr ARROW branch_body"""
        cond = cast(syntax.Expr, args[0])
        body = _find_expr(args[1:])
        return syntax.IfBranch(
            cond=cond, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def if_else_branch(self, meta: Meta, args: _Args) -> syntax.IfBranch:
        """if_else_branch: PIPE? "else" ARROW branch_body"""
        body = _find_expr(args)
        return syntax.IfBranch(
            cond=ELSE, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def if_expr(self, meta: Meta, args: _Args) -> syntax.If:
        branches = tuple(a for a in args if isinstance(a, syntax.IfBranch))
        return syntax.If(
            branches=branches,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: case_expr
    # ------------------------------------------------------------------

    def case_branch(self, meta: Meta, args: _Args) -> syntax.CaseBranch:
        """case_branch: pattern ARROW branch_body"""
        _pat_types = (
            syntax.WildcardPattern, syntax.LiteralPattern,
            syntax.VarPattern, syntax.ConstructorPattern,
        )
        pat = next(a for a in args if isinstance(a, _pat_types))
        body = _find_expr([a for a in args if not isinstance(a, _pat_types)])
        assert isinstance(pat, _pat_types)
        return syntax.CaseBranch(
            pattern=pat, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def case_expr(self, meta: Meta, args: _Args) -> syntax.Case:
        """case_expr: "case" or_expr "of" (PIPE case_branch)+"""
        subject = cast(syntax.Expr, args[0])
        branches = tuple(a for a in args[1:] if isinstance(a, syntax.CaseBranch))
        return syntax.Case(
            subject=subject, branches=branches,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: do_expr
    # ------------------------------------------------------------------

    def loop_bound(self, meta: Meta, args: _Args) -> int:
        """Extract N from a LOOP_BOUND token and validate it is positive."""
        tok = args[0]
        assert isinstance(tok, Token)
        n = int(str(tok))
        if n <= 0:
            raise AglSyntaxError(
                f"Loop bound must be a positive integer; got {n}.",
                span=_span_from_meta(meta),
            )
        return n

    def do_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """do_body: suite_expr | inline_seq — pass through the inner expr."""
        inner = _find_non_token(args)
        return cast(syntax.Expr, inner)

    def inline_seq(self, meta: Meta, args: _Args) -> syntax.Block:
        """inline_seq: inline_item (SEMICOLON inline_item)*

        Build a Block from the inline items (or_exprs and binders).
        """
        items = tuple(
            cast(syntax.Item, a)
            for a in args
            if a is not None and not isinstance(a, Token)
        )
        return syntax.Block(
            items=items,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def inline_item(self, meta: Meta, args: _Args) -> syntax.Item:
        """inline_item: binder | or_expr — transparent."""
        inner = _find_non_token(args)
        return cast(syntax.Item, inner)

    def do_expr(self, meta: Meta, args: _Args) -> syntax.Do:
        """do_expr: "do" loop_bound? do_body "until" or_expr"""
        # Grammar children (post-transform): int? (loop_bound), Expr (do_body),
        # Expr (until condition) — no Token or None placeholders.
        limit: int | None = None
        exprs: list[syntax.Expr] = []
        for a in args:
            if isinstance(a, int):
                limit = a
            else:
                exprs.append(cast(syntax.Expr, a))
        assert len(exprs) == 2, "do_expr: expected body and condition"
        body, condition = exprs[0], exprs[1]
        return syntax.Do(
            limit=limit, body=body, condition=condition,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: try_expr / catch_clause
    # ------------------------------------------------------------------

    def try_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """try_body: suite_expr | or_expr (SEMICOLON or_expr)*

        Returns a Block (for multiple items) or the single expr.
        """
        items = [a for a in args if a is not None and not isinstance(a, Token)]
        if len(items) == 1:
            return cast(syntax.Expr, items[0])
        # Multiple or_exprs: wrap in a Block.
        return syntax.Block(
            items=tuple(cast(syntax.Item, i) for i in items),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def catch_type_pattern(self, meta: Meta, args: _Args) -> tuple[str | None, str | None]:
        """catch_pattern: TYPE_NAME ("as" VAR_NAME)?"""
        type_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        bind_tok = next((a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None)
        exc_type = str(type_tok) if type_tok else None
        binding = str(bind_tok) if bind_tok else None
        return (exc_type, binding)

    def catch_wildcard_pattern(
        self, meta: Meta, args: _Args
    ) -> tuple[str | None, str | None]:
        """catch_pattern: VAR_NAME ("as" VAR_NAME)?  where first VAR_NAME is "_"."""
        var_toks = [a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"]
        wildcard_tok = var_toks[0]
        if str(wildcard_tok) != "_":
            raise AglSyntaxError(
                f"{str(wildcard_tok)!r} is not an exception type name. "
                "Catch patterns take an exception type (capitalized) or '_'.",
                span=_span_from_token(wildcard_tok),
            )
        binding: str | None = str(var_toks[1]) if len(var_toks) >= 2 else None
        return (None, binding)

    def catch_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """catch_body: suite_expr | or_expr — pass through the inner expr."""
        inner = _find_non_token(args)
        return cast(syntax.Expr, inner)

    def catch_clause(self, meta: Meta, args: _Args) -> syntax.CatchClause:
        """catch_clause: "catch" catch_pattern ARROW catch_body"""
        exc_type: str | None = None
        binding: str | None = None
        body: syntax.Expr | None = None
        for a in args:
            if isinstance(a, tuple):
                if len(a) == 2 and (a[0] is None or isinstance(a[0], str)) and (
                    a[1] is None or isinstance(a[1], str)
                ):
                    exc_type, binding = cast(tuple[str | None, str | None], a)
                elif a is not None:  # pragma: no cover
                    # Grammar guarantees tuples in catch_clause come only from
                    # catch_pattern (a (str|None, str|None) pair); no other
                    # tuple-valued child is possible.
                    body = cast(syntax.Expr, a)
            elif a is not None and not isinstance(a, Token):
                body = cast(syntax.Expr, a)
        assert body is not None, "catch_clause: no body"
        return syntax.CatchClause(
            exc_type=exc_type, binding=binding, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def try_expr(self, meta: Meta, args: _Args) -> syntax.Try:
        """try_expr: "try" try_body (catch_clause)+"""
        handlers = [a for a in args if isinstance(a, syntax.CatchClause)]
        try_body = next(
            a for a in args
            if a is not None and not isinstance(a, Token) and not isinstance(a, syntax.CatchClause)
        )
        return syntax.Try(
            body=cast(syntax.Expr, try_body),
            handlers=tuple(handlers),
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: raise_expr
    # ------------------------------------------------------------------

    def raise_expr(self, meta: Meta, args: _Args) -> syntax.Raise:
        exc = _find_expr(args)
        return syntax.Raise(exc=exc, span=_span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # suite_expr / branch_body
    # ------------------------------------------------------------------

    def suite_expr(self, meta: Meta, args: _Args) -> syntax.Block:
        """suite_expr: _INDENT block _DEDENT — unwrap to Block."""
        block = next(a for a in args if isinstance(a, syntax.Block))
        return block

    def branch_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """branch_body: suite_expr | or_expr — pass through the inner expr."""
        inner = _find_non_token(args)
        return cast(syntax.Expr, inner)

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def pat_var_or_wild(
        self, meta: Meta, args: _Args
    ) -> syntax.WildcardPattern | syntax.VarPattern:
        """VAR_NAME → WildcardPattern (when value is "_") or VarPattern."""
        tok = args[0]
        assert isinstance(tok, Token)
        if str(tok) == "_":
            return syntax.WildcardPattern(span=_span_from_meta(meta), node_id=self._next_id())
        return syntax.VarPattern(
            name=str(tok), span=_span_from_meta(meta), node_id=self._next_id()
        )

    def pat_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        type_toks = [a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"]
        qualifier: str | None = None
        if len(type_toks) == 2:
            qualifier = str(type_toks[0])
            name = str(type_toks[1])
        else:
            name = str(type_toks[0])
        fields: tuple[syntax.PatternField, ...] = ()
        for a in args:
            if isinstance(a, tuple) and all(isinstance(x, syntax.PatternField) for x in a):
                fields = cast(tuple[syntax.PatternField, ...], a)
        return syntax.ConstructorPattern(
            qualifier=qualifier, name=name, fields=fields,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def _literal_pattern(
        self,
        literal: syntax.IntLit | syntax.DecimalLit | syntax.BoolLit | syntax.StringLit
        | syntax.NullLit,
        meta: Meta,
    ) -> syntax.LiteralPattern:
        return syntax.LiteralPattern(
            literal=literal, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def pat_lit_int(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        lit = syntax.IntLit(
            value=int(str(tok)), span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_decimal(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        lit = syntax.DecimalLit(
            value=decimal.Decimal(str(tok)),
            span=_span_from_meta(meta), node_id=self._next_id(),
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_true(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(
            value=True, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_false(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(
            value=False, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_null(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.NullLit(span=_span_from_meta(meta), node_id=self._next_id())
        return self._literal_pattern(lit, meta)

    def pat_lit_str(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tmpl = _require_literal_string(
            args[0], "Pattern string literals cannot contain interpolation."
        )
        return self._literal_pattern(tmpl, meta)

    def pattern_fields(self, meta: Meta, args: _Args) -> tuple[syntax.PatternField, ...]:
        return tuple(a for a in args if isinstance(a, syntax.PatternField))

    def pat_field_full(self, meta: Meta, args: _Args) -> syntax.PatternField:
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None)
        assert name_tok is not None
        _pat_types = (
            syntax.WildcardPattern, syntax.LiteralPattern,
            syntax.VarPattern, syntax.ConstructorPattern,
        )
        pat = next((a for a in args if isinstance(a, _pat_types)), None)
        assert pat is not None
        assert isinstance(pat, _pat_types)
        return syntax.PatternField(
            name=str(name_tok), pattern=pat,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def pat_field_shorthand(self, meta: Meta, args: _Args) -> syntax.PatternField:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        name = str(name_tok)
        var_pat = syntax.VarPattern(
            name=name, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return syntax.PatternField(
            name=name, pattern=var_pat,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # template
    # ------------------------------------------------------------------

    def template(self, meta: Meta, args: _Args) -> syntax.Template | syntax.StringLit:
        """Build a Template from its segment children.

        Invariant: ``segments`` never contains empty ``TextSegment`` nodes.
        A template with no interpolation segments collapses to a ``StringLit``.
        """
        segments: list[syntax.TemplateSegment] = [
            a
            for a in args
            if isinstance(a, (syntax.TextSegment, syntax.InterpSegment))
            if not (isinstance(a, syntax.TextSegment) and a.text == "")
        ]
        span = _span_from_meta(meta)
        nid = self._next_id()
        if all(isinstance(s, syntax.TextSegment) for s in segments):
            text = "".join(s.text for s in segments if isinstance(s, syntax.TextSegment))
            return syntax.StringLit(value=text, span=span, node_id=nid)
        return syntax.Template(
            segments=tuple(segments),
            span=span,
            node_id=nid,
        )

    def tmpl_text(self, meta: Meta, args: _Args) -> syntax.TextSegment:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.TextSegment(
            text=str(tok),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def tmpl_interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        (seg,) = args
        assert isinstance(seg, syntax.InterpSegment)
        return seg

    def interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        # Grammar: INTERP_START expr INTERP_END
        expr: syntax.Expr = _find_expr(args)
        return syntax.InterpSegment(
            expr=expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # List and dict literals
    # ------------------------------------------------------------------

    def lit_list(self, meta: Meta, args: _Args) -> syntax.ListLit:
        """lit_list: LSQB (expr (COMMA expr)* COMMA?)? RSQB"""
        elements = tuple(
            cast(syntax.Expr, a)
            for a in args
            if a is not None and not isinstance(a, Token)
        )
        return syntax.ListLit(
            elements=elements,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_dict(self, meta: Meta, args: _Args) -> syntax.DictLit:
        """lit_dict: LBRACE (dict_entry (COMMA dict_entry)* COMMA?)? RBRACE"""
        entries = tuple(a for a in args if isinstance(a, syntax.DictEntry))
        return syntax.DictLit(
            entries=entries,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_str(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: template COLON expr — quoted string key."""
        non_tokens = [a for a in args if a is not None and not isinstance(a, Token)]
        assert len(non_tokens) >= 2, (
            f"dict_entry_str: expected key + expr, got {args!r}"
        )
        key_lit = _require_literal_string(
            non_tokens[0],
            "dict keys must be literal strings (no interpolation).",
        )
        val_expr = cast(syntax.Expr, non_tokens[1])
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_name(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: VAR_NAME COLON expr — identifier shorthand key."""
        name_tok = _find_name_token(args)
        val_expr = _find_expr(args[1:])
        key_lit = syntax.StringLit(
            value=str(name_tok),
            span=_span_from_token(name_tok),
            node_id=self._next_id(),
        )
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )


# ---------------------------------------------------------------------------
# Helper: find first TypeExpr in an args list
# ---------------------------------------------------------------------------


def _find_type_expr(args: _Args) -> TypeExpr:
    """Return the first element that is a TypeExpr instance."""
    for a in args:
        if isinstance(a, _ALL_TYPE_EXPRS):
            return a
    raise AssertionError(f"_find_type_expr: no TypeExpr found in {args!r}")  # pragma: no cover


def _find_name_token(args: _Args) -> Token:
    """Return the field/key name Token from a ``field_name``-bearing rule.

    ``field_name`` matches either a ``VAR_NAME`` identifier or the reserved word
    ``agent`` (token type ``AGENT``); both arrive here as plain name Tokens.
    """
    for a in args:
        if isinstance(a, Token) and a.type in ("VAR_NAME", "AGENT"):
            return a
    raise AssertionError(f"_find_name_token: no name token found in {args!r}")  # pragma: no cover


def _require_literal_string(node: object, message: str) -> syntax.StringLit:
    """Return *node* as a ``StringLit``, rejecting an interpolated ``Template``."""
    if isinstance(node, syntax.StringLit):
        return node
    assert isinstance(node, syntax.Template), (
        f"_require_literal_string: expected StringLit or Template, got {type(node)}"
    )
    raise AglSyntaxError(message, span=node.span)


def _is_expr_obj(a: object) -> bool:
    """Return True if *a* is an Expr (AST node, not a Token or None)."""
    return a is not None and not isinstance(a, Token)


def _is_expr_node(a: object) -> bool:
    """Return True if *a* is an expression AST node."""
    return isinstance(a, syntax.Expr)


def _is_set_target(a: object) -> bool:
    """Return True if *a* is an assignment target AST node."""
    return isinstance(a, (syntax.NameTarget, syntax.IndexTarget))


def _find_non_token(args: _Args) -> object:
    """Return the first non-None, non-Token element in args."""
    result = next(
        (a for a in args if a is not None and not isinstance(a, Token)),
        None,
    )
    if result is None:  # pragma: no cover
        raise AssertionError(f"_find_non_token: no non-token found in {args!r}")
    return result


def _find_expr(args: _Args) -> syntax.Expr:
    """Return the first Expr in *args* (skip Tokens and None placeholders).

    Also used to extract single-Expr branch/suite bodies, which are likewise
    the sole non-token element in *args*.
    """
    return cast(syntax.Expr, _find_non_token(args))


def _extract_ann_and_value(
    tail: _Args,
) -> tuple[TypeExpr | None, syntax.Expr]:
    """Extract (type_ann, value) from the tail of a let/var args list.

    Grammar tail is: type_ann? EQ expr
    With maybe_placeholders=True: [None|TypeExpr, Token(EQ), Expr]
    """
    ann: TypeExpr | None = None
    value: syntax.Expr | None = None
    for a in tail:
        if isinstance(a, _ALL_TYPE_EXPRS):
            ann = a
        elif _is_expr_obj(a) and not isinstance(a, Token):
            value = cast(syntax.Expr, a)
    assert value is not None, f"_extract_ann_and_value: no Expr found in {tail!r}"
    return ann, value


def _extract_ann_and_optional_expr(
    tail: _Args,
) -> tuple[TypeExpr | None, syntax.Expr | None]:
    """Extract (type_ann, optional_expr) from a param declaration tail."""
    ann: TypeExpr | None = None
    value: syntax.Expr | None = None
    for a in tail:
        if isinstance(a, _ALL_TYPE_EXPRS):
            ann = a
        elif _is_expr_obj(a):
            value = cast(syntax.Expr, a)
    return ann, value


def syntax_error_from_meta(meta: Meta, message: str) -> AglSyntaxError:
    """Create an AglSyntaxError from a Meta object."""
    return AglSyntaxError(message, span=_span_from_meta(meta))
