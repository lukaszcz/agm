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
from dataclasses import dataclass, replace
from itertools import count
from typing import Iterable, Mapping, TypeAlias, TypeGuard, cast

from lark import Transformer, v_args
from lark.lexer import Token
from lark.tree import Meta

import agm.agl.syntax as syntax
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.syntax.nodes import ELSE
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceId, SourceSpan
from agm.agl.syntax.types import (
    AgentT,
    AppliedT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    ImportMode,
    IntT,
    JsonT,
    ListT,
    NameT,
    Qualifier,
    TextT,
    TypeExpr,
    TypeQualifier,
    UnitT,
)

# Types used internally
_NamedArgList = list[syntax.NamedArg]


@dataclass(frozen=True, slots=True)
class _RawPlaceholder:
    """Transformer-internal placeholder argument before call-level validation."""

    raw_digits: str | None
    span: SourceSpan


# ---------------------------------------------------------------------------
# Transformer-internal marker sentinel (never leaks into the AST)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParamMarker:
    """Transformer-internal sentinel for a zone-boundary marker in a param/field list.

    Exists only during transformation; the AST never contains these.
    ``zone`` is the zone the marker *opens*; ``label`` is the source spelling
    (``"/"``, ``"*"``, ``"@pos"``, etc.) for error messages.
    """

    zone: syntax.ParamKind
    label: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _InfixOperator:
    """Transformer-internal operator token in a flat infix chain."""

    name: str
    builtin: syntax.BinOp | None
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _InfixPriority:
    """Transformer-internal priority specification for ``infix`` declarations."""

    value: int | None
    base: str | None
    delta: int


@dataclass(frozen=True, slots=True)
class _InfixOperand:
    """Transformer-internal infix operand with pending prefix ``not`` operators."""

    expr: syntax.Expr | _RawInfixChain
    not_count: int
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _RawInfixChain:
    """Transformer-internal flat chain awaiting declaration-aware regrouping."""

    operands: tuple[_InfixOperand, ...]
    operators: tuple[_InfixOperator, ...]
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _RawNamedArg:
    """Transformer-internal named argument whose value may be a placeholder."""

    name: str
    value: _RawPlaceholder
    span: SourceSpan


_RawPosArg: TypeAlias = syntax.Expr | _RawPlaceholder | _RawInfixChain
_RawNamed: TypeAlias = syntax.NamedArg | _RawNamedArg
_RawArgLists: TypeAlias = tuple[list[_RawPosArg], list[_RawNamed]]
_ArgLists: TypeAlias = tuple[list[syntax.Expr], list[syntax.NamedArg]]
_JuxtCall: TypeAlias = tuple[tuple[TypeExpr, ...], _ArgLists]
_JuxtSuffix: TypeAlias = tuple[str, str] | tuple[str, syntax.Expr] | tuple[str, _JuxtCall]
_RawItem: TypeAlias = syntax.Item | _RawInfixChain


# Zone ordering for marker validation (strictly increasing).
_ZONE_ORDER: dict[syntax.ParamKind, int] = {
    syntax.ParamKind.POSITIONAL_ONLY: 0,
    syntax.ParamKind.STANDARD: 1,
    syntax.ParamKind.NAMED_ONLY: 2,
}
_ZONE_BY_ORDER: dict[int, syntax.ParamKind] = {v: k for k, v in _ZONE_ORDER.items()}

# Zone opened by each `@`-marker name (validated in marker_at).
_AT_ZONE: dict[str, syntax.ParamKind] = {
    "pos": syntax.ParamKind.POSITIONAL_ONLY,
    "std": syntax.ParamKind.STANDARD,
    "named": syntax.ParamKind.NAMED_ONLY,
}

# Interleaved sequence type produced by field_list / param_list.
_RawEntries: TypeAlias = tuple[syntax.Param | _ParamMarker, ...]


@dataclass(frozen=True, slots=True)
class _PatternFieldsSplit:
    """Transformer-internal split of pattern_fields into positional and named.

    Exists only during transformation; the AST receives the split as two
    separate fields on ``ConstructorPattern``.
    """

    positional: tuple[syntax.Pattern, ...]
    named: tuple[syntax.PatternField, ...]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Args = list[object]  # Rule children after transformation (tokens + AST nodes)
_NAME_TOKEN_TYPES = frozenset({"NAME", "OP_NAME"})

_ALL_TYPE_EXPRS = (
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
)

# The concrete pattern AST node types (used to pick a sub-pattern out of rule children).
_PATTERN_NODE_TYPES = (
    syntax.WildcardPattern,
    syntax.LiteralPattern,
    syntax.VarPattern,
    syntax.ConstructorPattern,
)

_BUILTIN_INFIX_PRIORITIES: dict[str, int] = {
    "or": 10,
    "and": 20,
    "in": 30,
    "==": 30,
    "!=": 30,
    "<": 30,
    "<=": 30,
    ">": 30,
    ">=": 30,
    "+": 40,
    "-": 40,
    "*": 50,
    "/": 50,
}
_BUILTIN_INFIX_ASSOC: dict[str, syntax.InfixAssoc] = {
    name: syntax.InfixAssoc.LEFT for name in _BUILTIN_INFIX_PRIORITIES
}
_BUILTIN_INFIX_OPS: dict[str, syntax.BinOp] = {op.value: op for op in syntax.BinOp}
_NON_ASSOC_INFIX: frozenset[str] = frozenset({"in", "==", "!=", "<", "<=", ">", ">="})
_DEFAULT_USER_INFIX_PRIORITY = 40
_NOT_PRIORITY = 25


def _is_str_tuple(a: object) -> bool:
    """Return True iff *a* is a non-empty tuple whose elements are all ``str``."""
    return isinstance(a, tuple) and len(a) > 0 and all(isinstance(x, str) for x in a)


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


# ---------------------------------------------------------------------------
# AstBuilder
# ---------------------------------------------------------------------------


@v_args(meta=True)
class AstBuilder(Transformer):
    """Transforms a Lark parse tree into ``agm.agl.syntax`` dataclasses.

    The ``node_id`` counter is monotonically increasing and starts at
    ``start_id`` (default ``0``).  Each ``parse_program`` call creates a fresh
    builder, so node IDs within a module source are deterministic (assigned
    in tree-walk order — root first, depth-first left-to-right).  Incremental
    sessions seed ``start_id`` from a prior parse's ``next_node_id`` so ids
    stay globally unique across entries.
    """

    def __init__(
        self,
        *,
        start_id: int = 0,
        source: SourceId | None = None,
        ambient_infix: "Mapping[str, tuple[int, syntax.InfixAssoc]] | None" = None,
    ) -> None:
        super().__init__()
        self._counter = count(start_id)
        # The next id the counter will hand out.  Tracked explicitly so callers
        # can read the first id NOT consumed after a transform (the seed for a
        # subsequent incremental parse) without having to assume the root node
        # holds the maximum id.  Seeded to ``start_id`` before any node is built.
        self._next_unused: int = start_id
        # Source identity stamped on every span this builder constructs.
        # Defaults to UNKNOWN_SOURCE when no source is supplied.
        self._source: SourceId = source if source is not None else UNKNOWN_SOURCE
        # Already-resolved user infix fixity carried over from a prior context
        # (REPL entries). Merged into the operator table so an operator declared
        # in an earlier entry can be used in a later one. ``None`` for a standalone
        # whole-program parse.
        self._ambient_infix = ambient_infix

    def _span_from_meta(self, meta: Meta) -> SourceSpan:
        """Build a SourceSpan from Lark tree Meta, stamped with self._source."""
        return SourceSpan(
            start_line=meta.line,
            start_col=meta.column,
            end_line=meta.end_line,
            end_col=meta.end_column,
            start_offset=meta.start_pos,
            end_offset=meta.end_pos,
            source=self._source,
        )

    def _span_from_token(self, tok: Token) -> SourceSpan:
        """Build a SourceSpan from a Lark Token's position fields, stamped with self._source."""
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
            source=self._source,
        )

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
        span = self._span_from_meta(meta)
        table = _operator_table_from_decls(block.items, self._ambient_infix)
        body = _rewrite_block_infix(block, table, self)
        return syntax.Program(body=body, span=span, node_id=self._next_id())

    def block(self, meta: Meta, args: _Args) -> syntax.Block:
        """block: item (_sep item)* _sep?

        _NEWLINE tokens are filtered by leading underscore convention.
        SEMICOLON tokens are %declare'd and appear in tree — drop them.
        items are Declaration | Binder | Expr (all are AST nodes, not Tokens).
        """
        items = tuple(cast(_RawItem, a) for a in args if a is not None and not isinstance(a, Token))
        return syntax.Block(
            items=cast(tuple[syntax.Item, ...], items),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def param_decl(self, meta: Meta, args: _Args) -> syntax.ParamDecl:
        # Grammar: "param" name type_ann? (EQ expr)?
        name_tok = _find_name_token(args)
        ann, default = _extract_ann_and_optional_expr(args[1:])
        span = self._span_from_meta(meta)
        return syntax.ParamDecl(
            name=str(name_tok),
            annotation=ann,
            default=default,
            span=span,
            node_id=self._next_id(),
        )

    def program_decl(self, meta: Meta, args: _Args) -> syntax.ProgramDecl:
        # Grammar: "program" name
        name_tok = _find_name_token(args)
        span = self._span_from_meta(meta)
        return syntax.ProgramDecl(
            name=str(name_tok),
            span=span,
            node_id=self._next_id(),
        )

    def agent_decl(self, meta: Meta, args: _Args) -> syntax.AgentDecl:
        # Grammar: AGENT name (EQ template)?
        name_tok = next(a for a in args if _is_name_token(a))
        runner_node = next(
            (a for a in args if isinstance(a, (syntax.StringLit, syntax.Template))),
            None,
        )
        runner: str | None = (
            None
            if runner_node is None
            else _require_literal_string(
                runner_node,
                "agent runner string must be a literal string with no interpolation.",
            ).value
        )
        span = self._span_from_meta(meta)
        return syntax.AgentDecl(
            name=str(name_tok),
            runner=runner,
            span=span,
            node_id=self._next_id(),
        )

    def builtin_var_def(self, meta: Meta, args: _Args) -> syntax.BuiltinVarDecl:
        """builtin_var_def: "builtin" _NEWLINE? "var" name type_ann"""
        name_tok = _find_name_token(args)
        type_expr = _find_type_expr(args[1:])
        return syntax.BuiltinVarDecl(
            name=str(name_tok),
            type_ann=type_expr,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _make_infix_decl(
        self, meta: Meta, args: _Args, assoc: syntax.InfixAssoc
    ) -> syntax.InfixDecl:
        op = next(a for a in args if isinstance(a, _InfixOperator))
        priority = next((a for a in args if isinstance(a, _InfixPriority)), None)
        return syntax.InfixDecl(
            name=op.name,
            assoc=assoc,
            priority=None if priority is None else priority.value,
            priority_base=None if priority is None else priority.base,
            priority_delta=0 if priority is None else priority.delta,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def infixl_decl(self, meta: Meta, args: _Args) -> syntax.InfixDecl:
        """infix_decl: "infixl" infix_op infix_priority?"""
        return self._make_infix_decl(meta, args, syntax.InfixAssoc.LEFT)

    def infixr_decl(self, meta: Meta, args: _Args) -> syntax.InfixDecl:
        """infix_decl: "infixr" infix_op infix_priority?"""
        return self._make_infix_decl(meta, args, syntax.InfixAssoc.RIGHT)

    def infix_priority_literal(self, meta: Meta, args: _Args) -> _InfixPriority:
        self._require_infix_at_keyword(meta, args)
        tok = next(a for a in args if isinstance(a, Token) and a.type == "INT")
        return _InfixPriority(value=int(str(tok)), base=None, delta=0)

    def infix_priority_relative(self, meta: Meta, args: _Args) -> _InfixPriority:
        self._require_infix_at_keyword(meta, args)
        op = next(a for a in args if isinstance(a, _InfixOperator))
        delta = next((a for a in args if isinstance(a, int)), 0)
        return _InfixPriority(value=None, base=op.name, delta=delta)

    def _require_infix_at_keyword(self, meta: Meta, args: _Args) -> None:
        first_name = next((a for a in args if isinstance(a, Token) and a.type == "NAME"), None)
        if first_name is None or str(first_name) != "at":
            raise syntax_error_from_meta(meta, "infix priority must start with 'at'.")

    def priority_delta_plus(self, meta: Meta, args: _Args) -> int:
        tok = next(a for a in args if isinstance(a, Token) and a.type == "INT")
        return int(str(tok))

    def priority_delta_minus(self, meta: Meta, args: _Args) -> int:
        tok = next(a for a in args if isinstance(a, Token) and a.type == "INT")
        return -int(str(tok))

    # ------------------------------------------------------------------
    # record_def / field_def
    # ------------------------------------------------------------------

    def record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        # Grammar: "record" name type_params? EQ? record_body
        name_tok = _find_name_token(args)
        type_params_val: tuple[str, ...] = ()
        for a in args:
            if _is_str_tuple(a):
                type_params_val = cast(tuple[str, ...], a)
        fields = _find_field_tuple(args)
        return syntax.RecordDef(
            name=str(name_tok),
            fields=fields,
            type_params=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Marker transformer methods (grammar aliases → _ParamMarker sentinels)
    # ------------------------------------------------------------------

    def marker_slash(self, meta: Meta, args: _Args) -> _ParamMarker:
        """SLASH → _ParamMarker(STANDARD) — the '/' pos-only→standard boundary."""
        return _ParamMarker(
            zone=syntax.ParamKind.STANDARD,
            label="/",
            span=self._span_from_meta(meta),
        )

    def marker_star(self, meta: Meta, args: _Args) -> _ParamMarker:
        """STAR → _ParamMarker(NAMED_ONLY) — the '*' standard→named-only boundary."""
        return _ParamMarker(
            zone=syntax.ParamKind.NAMED_ONLY,
            label="*",
            span=self._span_from_meta(meta),
        )

    def marker_at(self, meta: Meta, args: _Args) -> _ParamMarker:
        """AT NAME → _ParamMarker; NAME must be pos/std/named (else AglSyntaxError)."""
        name_tok = next(a for a in args if _is_name_token(a))
        label_name = str(name_tok)
        span = self._span_from_meta(meta)
        zone = _AT_ZONE.get(label_name)
        if zone is None:
            raise AglSyntaxError(
                f"unknown parameter marker '@{label_name}'; valid markers are @pos, @std, @named.",
                span=span,
            )
        return _ParamMarker(zone=zone, label=f"@{label_name}", span=span)

    # ------------------------------------------------------------------
    # record_def / field_def (continued)
    # ------------------------------------------------------------------

    def record_indent_body(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: param_marker? _INDENT block_entry (_NEWLINE block_entry)* _NEWLINE? _DEDENT
        # block_entry is ?field_def | ?param_marker — collect all in order, then resolve.
        # Records are always named-only by default.
        entries: _RawEntries = tuple(a for a in args if isinstance(a, (syntax.Param, _ParamMarker)))
        return _resolve_params(entries, default_kind=syntax.ParamKind.NAMED_ONLY)

    def record_paren_body(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: LPAR field_list? RPAR
        # field_list returns _RawEntries; resolve with named-only default.
        for a in args:
            if _is_field_tuple(a):
                return _resolve_params(
                    cast(_RawEntries, a), default_kind=syntax.ParamKind.NAMED_ONLY
                )
        return ()

    record_inline_body = record_paren_body

    def field_def(self, meta: Meta, args: _Args) -> syntax.Param:
        # Grammar: field_name COLON type_expr
        # Build with a provisional NAMED_ONLY kind; the owner builder
        # (record_indent_body, record_paren_body, variant_payload, exception bodies)
        # reassigns the kind via _resolve_params().
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        type_expr = _find_type_expr(args[1:])
        return syntax.Param(
            name=str(name_tok),
            type_expr=type_expr,
            kind=syntax.ParamKind.NAMED_ONLY,
            default=None,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # enum_def / variant_def / variant_payload / field_list / field_inline
    # ------------------------------------------------------------------

    def enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        # Grammar: "enum" name type_params? EQ? enum_body
        name_tok = _find_name_token(args)
        type_params_val: tuple[str, ...] = ()
        for a in args:
            if _is_str_tuple(a):
                type_params_val = cast(tuple[str, ...], a)
        variants = _find_variant_tuple(args)
        return syntax.EnumDef(
            name=str(name_tok),
            variants=variants,
            type_params=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def enum_body(self, meta: Meta, args: _Args) -> tuple[syntax.VariantDef, ...]:
        return _find_variant_tuple(args)

    def enum_variant_seq(self, meta: Meta, args: _Args) -> tuple[syntax.VariantDef, ...]:
        return tuple(a for a in args if isinstance(a, syntax.VariantDef))

    def variant_def(self, meta: Meta, args: _Args) -> syntax.VariantDef:
        # Grammar: PIPE? name variant_payload?
        name_tok = next(
            (a for a in args if _is_name_token(a)),
            None,
        )
        assert name_tok is not None, "variant_def: no name token"
        fields: tuple[syntax.Param, ...] = ()
        for a in args:
            if isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], syntax.Param)):
                fields = cast(tuple[syntax.Param, ...], a)
                break
        return syntax.VariantDef(
            name=str(name_tok),
            fields=fields,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def variant_payload(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: LPAR field_list? RPAR
        # field_list returns _RawEntries; count only Param entries for the
        # single-field→STANDARD default; markers don't count as fields.
        for a in args:
            if _is_field_tuple(a):
                raw = cast(_RawEntries, a)
                param_count = sum(1 for x in raw if isinstance(x, syntax.Param))
                default_kind = (
                    syntax.ParamKind.STANDARD if param_count == 1 else syntax.ParamKind.NAMED_ONLY
                )
                return _resolve_params(raw, default_kind=default_kind)
        return ()

    def field_list(self, meta: Meta, args: _Args) -> _RawEntries:
        # Grammar: field_entry (COMMA field_entry)* COMMA?
        # ?field_entry is transparent: Param (from field_inline/field_def) and
        # _ParamMarker (from param_marker) arrive directly as children.
        # Return the raw interleaving; zone resolution happens in the owning builder.
        return tuple(a for a in args if isinstance(a, (syntax.Param, _ParamMarker)))

    # Grammar: field_name COLON type_expr — identical shape to ``field_def``.
    field_inline = field_def

    # ------------------------------------------------------------------
    # exception_def / exception_body
    # ------------------------------------------------------------------

    def exception_def(self, meta: Meta, args: _Args) -> syntax.ExceptionDef:
        # Grammar: "exception" name exception_base? exception_body
        name_tok = _find_name_token(args)
        base = next((a for a in args if type(a) is str), None)
        fields = _find_field_tuple(args)
        return syntax.ExceptionDef(
            name=str(name_tok),
            fields=fields,
            base=base,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def exception_base(self, meta: Meta, args: _Args) -> str:
        name_tok = _find_name_token(args)
        return str(name_tok)

    exception_indent_body = record_indent_body
    exception_paren_body = record_paren_body
    exception_inline_body = record_paren_body

    # ------------------------------------------------------------------
    # type_alias
    # ------------------------------------------------------------------

    def type_alias(self, meta: Meta, args: _Args) -> syntax.TypeAlias:
        # Grammar: "type" name type_params? EQ type_expr
        name_tok = _find_name_token(args)
        type_params_val: tuple[str, ...] = ()
        for a in args:
            if _is_str_tuple(a):
                type_params_val = cast(tuple[str, ...], a)
        type_expr = _find_type_expr(args)
        return syntax.TypeAlias(
            name=str(name_tok),
            type_expr=type_expr,
            type_params=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # func_def / param_list / param_def / func_body
    # ------------------------------------------------------------------

    def func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """func_def: "def" name type_params? LPAR param_list? RPAR
        (THIN_ARROW type_expr)? (EQ func_body | suite_expr)
        """
        name_tok = _find_name_token(args)
        type_params_val: tuple[str, ...] = ()
        for a in args:
            if _is_str_tuple(a):
                type_params_val = cast(tuple[str, ...], a)
        params, return_type, body = self._split_params_type_body(args)
        assert body is not None, "func_def: no body"
        return syntax.FuncDef(
            name=str(name_tok),
            params=params,
            return_type=return_type,
            body=body,
            type_params=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _bodyless_func_def(
        self, meta: Meta, args: _Args, *, is_builtin: bool = False, is_extern: bool = False
    ) -> syntax.FuncDef:
        """Shared construction for body-less function signatures.

        Both ``builtin_func_def`` and ``extern_func_def`` share the shape
        "def" name type_params? (params) -> type_expr with no body; only the
        leading modifier and the resulting flag differ.
        """
        name_tok = _find_name_token(args)
        type_params_val: tuple[str, ...] = ()
        for a in args:
            if _is_str_tuple(a):
                type_params_val = cast(tuple[str, ...], a)
        params, return_type, body = self._split_params_type_body(args)
        assert return_type is not None, "bodyless func def: no return type"
        assert body is None, "bodyless func def: unexpected body"
        return syntax.FuncDef(
            name=str(name_tok),
            params=params,
            return_type=return_type,
            body=None,
            type_params=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_builtin=is_builtin,
            is_extern=is_extern,
        )

    def builtin_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """builtin_func_def: "builtin" "def" name type_params? (...) -> type_expr"""
        return self._bodyless_func_def(meta, args, is_builtin=True)

    def extern_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """extern_func_def: "extern" "def" name type_params? (...) -> type_expr"""
        return self._bodyless_func_def(meta, args, is_extern=True)

    def param_list(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        """param_list: param_entry (COMMA param_entry)* COMMA?

        Collects the full marker/param interleaving and resolves zones.
        def/lambda parameters default to STANDARD when no markers are present.
        """
        entries: _RawEntries = tuple(a for a in args if isinstance(a, (syntax.Param, _ParamMarker)))
        return _resolve_params(entries, default_kind=syntax.ParamKind.STANDARD)

    def param_def(self, meta: Meta, args: _Args) -> syntax.Param:
        """param_def: field_name COLON type_expr (EQ or_expr)?"""
        name_tok = _find_name_token(args)
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
            kind=syntax.ParamKind.STANDARD,
            default=default,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def func_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """func_body: suite_expr | func_inline_seq | expr — pass through the inner expr."""
        # args[0] is a Block (from suite_expr/func_inline_seq) or any Expr (from expr).
        expr = _find_non_token(args)
        return cast(syntax.Expr, expr)

    def func_inline_seq(self, meta: Meta, args: _Args) -> syntax.Block:
        """func_inline_seq: binder (SEMICOLON binder)* SEMICOLON expr."""
        items = tuple(
            cast(syntax.Item, a) for a in args if a is not None and not isinstance(a, Token)
        )
        return syntax.Block(
            items=items,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # let_decl / var_decl / assign_stmt
    # ------------------------------------------------------------------

    def let_decl(self, meta: Meta, args: _Args) -> syntax.LetDecl:
        # Grammar: "let" name type_ann? EQ expr
        name_tok = _find_name_token(args)
        ann, value = _extract_ann_and_value(args[1:])
        span = self._span_from_meta(meta)
        return syntax.LetDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def var_decl(self, meta: Meta, args: _Args) -> syntax.VarDecl:
        # Grammar: "var" name type_ann? EQ expr
        name_tok = _find_name_token(args)
        ann, value = _extract_ann_and_value(args[1:])
        span = self._span_from_meta(meta)
        return syntax.VarDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def assign_stmt(self, meta: Meta, args: _Args) -> syntax.AssignStmt:
        # Grammar: postfix ASSIGN expr
        lhs, value = (cast(syntax.Expr, a) for a in args if _is_expr_node(a))
        root = lhs
        while isinstance(root, syntax.IndexAccess):
            root = root.obj
        if not isinstance(root, syntax.VarRef) or root.type_qualifier is not None:
            raise AglSyntaxError(
                "assignment target must be a variable or indexed variable.",
                span=lhs.span,
            )
        if root.module_qualifier is not None and isinstance(lhs, syntax.IndexAccess):
            raise AglSyntaxError(
                "assignment target cannot combine a module qualifier with indexing; "
                "a qualified assignment target must be a bare name.",
                span=lhs.span,
            )
        if isinstance(lhs, syntax.VarRef):
            target: syntax.AssignTarget = syntax.NameTarget(
                name=lhs.name,
                span=lhs.span,
                node_id=self._next_id(),
                module_qualifier=lhs.module_qualifier,
            )
        else:
            assert isinstance(lhs, syntax.IndexAccess)
            target = syntax.IndexTarget(
                obj=lhs.obj,
                index=lhs.index,
                span=lhs.span,
                node_id=self._next_id(),
            )
        span = self._span_from_meta(meta)
        return syntax.AssignStmt(
            target=target,
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

    def prim_or_name(self, meta: Meta, args: _Args) -> TypeExpr:
        """name in type position — map NAME to primitive or NameT."""
        tok = args[0]
        assert isinstance(tok, Token)
        name = str(tok)
        span = self._span_from_meta(meta)
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

    def applied_type(self, meta: Meta, args: _Args) -> TypeExpr:
        """name type_lsqb type_arg_list RSQB — applied generic type."""
        name_tok = next(a for a in args if _is_name_token(a))
        name = str(name_tok)
        type_args: tuple[TypeExpr, ...] = cast(
            tuple[TypeExpr, ...],
            next(
                (
                    a
                    for a in args
                    if isinstance(a, tuple) and len(a) > 0 and isinstance(a[0], _ALL_TYPE_EXPRS)
                ),
                (),
            ),
        )
        span = self._span_from_meta(meta)
        nid = self._next_id()
        if name == "list":
            if len(type_args) == 1:
                return ListT(elem=type_args[0], span=span, node_id=nid)
            raise syntax_error_from_meta(meta, "list[] takes exactly one type argument")
        if name == "dict":
            if len(type_args) == 2:
                key_type = type_args[0]
                if not isinstance(key_type, TextT):
                    raise AglSyntaxError(
                        f"dict keys are always text in AgL, got {_type_expr_spelling(key_type)!r}.",
                        span=key_type.span,
                    )
                return DictT(value=type_args[1], span=span, node_id=nid)
            raise syntax_error_from_meta(meta, "dict[] takes exactly two type arguments")
        return AppliedT(name=name, args=type_args, span=span, node_id=nid)

    def qual_applied_type(self, meta: Meta, args: _Args) -> AppliedT:
        """qual_prefix name type_lsqb type_arg_list RSQB — qualified application."""
        qual = next(a for a in args if isinstance(a, Qualifier))
        name_tok = next(a for a in args if _is_name_token(a))
        type_args = cast(
            tuple[TypeExpr, ...],
            next(
                (
                    a
                    for a in args
                    if isinstance(a, tuple) and len(a) > 0 and isinstance(a[0], _ALL_TYPE_EXPRS)
                ),
                (),
            ),
        )
        return AppliedT(
            name=str(name_tok),
            args=type_args,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=qual,
        )

    def agent_type(self, meta: Meta, args: _Args) -> AgentT:
        """AGENT terminal in type position → AgentT."""
        return AgentT(span=self._span_from_meta(meta), node_id=self._next_id())

    def type_arg_list(self, meta: Meta, args: _Args) -> tuple[TypeExpr, ...]:
        """type_arg_list: type_expr (COMMA type_expr)*"""
        return tuple(a for a in args if isinstance(a, _ALL_TYPE_EXPRS))

    def type_params(self, meta: Meta, args: _Args) -> tuple[str, ...]:
        """type_params: type_lsqb type_param_list RSQB"""
        return cast(
            tuple[str, ...],
            next(
                (
                    a
                    for a in args
                    if isinstance(a, tuple) and len(a) > 0 and all(isinstance(x, str) for x in a)
                ),
                (),
            ),
        )

    def type_param_list(self, meta: Meta, args: _Args) -> tuple[str, ...]:
        """type_param_list: name (COMMA name)*"""
        return tuple(str(a) for a in args if _is_name_token(a))

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
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def unary_func_type(self, meta: Meta, args: _Args) -> FuncT:
        """type_atom THIN_ARROW type_expr — function type A -> B."""
        type_nodes = [a for a in args if isinstance(a, _ALL_TYPE_EXPRS)]
        assert len(type_nodes) == 2, "unary_func_type: expected parameter and result types"
        param_type, result_type = type_nodes
        return FuncT(
            params=(param_type,),
            result=result_type,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def type_list(self, meta: Meta, args: _Args) -> tuple[TypeExpr, ...]:
        """type_list: type_expr (COMMA type_expr)* COMMA?"""
        return tuple(a for a in args if isinstance(a, _ALL_TYPE_EXPRS))

    # ------------------------------------------------------------------
    # unit_lit / paren_expr
    # ------------------------------------------------------------------

    def unit_lit(self, meta: Meta, args: _Args) -> syntax.UnitLit:
        return syntax.UnitLit(span=self._span_from_meta(meta), node_id=self._next_id())

    def paren_expr(self, meta: Meta, args: _Args) -> syntax.Expr | _RawInfixChain:
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
            if _is_str_tuple(a):
                pass  # type_params: non-empty tuple of str — skip
            elif isinstance(a, tuple) and all(isinstance(x, syntax.Param) for x in a):
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
            span=self._span_from_meta(meta),
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
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_decimal(self, meta: Meta, args: _Args) -> syntax.DecimalLit:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.DecimalLit(
            value=decimal.Decimal(str(tok)),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_true(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(value=True, span=self._span_from_meta(meta), node_id=self._next_id())

    def lit_false(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(value=False, span=self._span_from_meta(meta), node_id=self._next_id())

    def lit_null(self, meta: Meta, args: _Args) -> syntax.NullLit:
        return syntax.NullLit(span=self._span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # var_ref / constructor
    # ------------------------------------------------------------------

    def var_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.VarRef(
            name=str(tok), span=self._span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # Postfix: call / field_access / index_access
    # ------------------------------------------------------------------

    def call(self, meta: Meta, args: _Args) -> syntax.Call:
        """postfix LPAR arg_list? RPAR → Call node."""
        callee = cast(syntax.Expr, args[0])
        type_args: tuple[TypeExpr, ...] = ()
        if isinstance(callee, syntax.TypeApply):
            type_args = callee.type_args
            callee = callee.expr
        raw_pos_args: list[_RawPosArg] = []
        raw_named_args: list[_RawNamed] = []
        for a in args[1:]:
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], list):
                pa, na = cast(_RawArgLists, a)
                raw_pos_args = pa
                raw_named_args = na
            # Tokens (LPAR, RPAR) and None are skipped

        span = self._span_from_meta(meta)
        pos_args, named_args = self._finalize_call_args(
            raw_pos_args, raw_named_args, call_span=span
        )
        return syntax.Call(
            callee=callee,
            args=pos_args,
            named_args=named_args,
            span=span,
            node_id=self._next_id(),
            type_args=type_args,
        )

    def field_access(self, meta: Meta, args: _Args) -> syntax.FieldAccess:
        """postfix DOT name — record field access."""
        obj_expr = cast(syntax.Expr, args[0])
        field_tok = _find_name_token(args)
        return syntax.FieldAccess(
            obj=obj_expr,
            field=str(field_tok),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def index_access(self, meta: Meta, args: _Args) -> syntax.IndexAccess:
        """postfix INDEX_LSQB expr RSQB — list/dict index access."""
        exprs = [a for a in args if _is_expr_node(a)]
        obj_expr, index_expr = exprs
        return syntax.IndexAccess(
            obj=cast(syntax.Expr, obj_expr),
            index=cast(syntax.Expr, index_expr),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
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
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def juxt_arg(self, meta: Meta, args: _Args) -> syntax.Expr:
        """juxt_arg: juxt_atom juxt_suffix*

        Builds the restricted postfix chain allowed by single-arg call sugar,
        such as ``print res.stdout``, ``print xs[0]``, and
        ``f Opt::Some(x = 1)`` and ``f Opt[int]::None()``.
        """
        non_tokens = [a for a in args if a is not None and not isinstance(a, Token)]
        assert non_tokens, "juxt_arg: no base atom"
        result = cast(syntax.Expr, non_tokens[0])
        for suffix_obj in non_tokens[1:]:
            kind, value = cast(_JuxtSuffix, suffix_obj)
            if kind == "field":
                result = syntax.FieldAccess(
                    obj=result,
                    field=cast(str, value),
                    span=self._span_from_meta(meta),
                    node_id=self._next_id(),
                )
            elif kind == "index":
                result = syntax.IndexAccess(
                    obj=result,
                    index=cast(syntax.Expr, value),
                    span=self._span_from_meta(meta),
                    node_id=self._next_id(),
                )
            else:
                type_args_val, arg_lists = cast(_JuxtCall, value)
                pos_args, named_args = arg_lists
                result = syntax.Call(
                    callee=result,
                    args=tuple(pos_args),
                    named_args=tuple(named_args),
                    type_args=type_args_val,
                    span=self._span_from_meta(meta),
                    node_id=self._next_id(),
                )
        return result

    def juxt_field_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_suffix: DOT field_name -> juxt_field_suffix."""
        field_tok = _find_name_token(args)
        return ("field", str(field_tok))

    def juxt_index_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_suffix: INDEX_LSQB expr RSQB -> juxt_index_suffix."""
        index_expr = cast(syntax.Expr, next(a for a in args if _is_expr_node(a)))
        return ("index", index_expr)

    def _juxt_finalized_arg_lists(self, meta: Meta, args: _Args) -> _ArgLists:
        """Validate and finalize the raw arg-list under a juxtaposition call suffix."""
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], list):
                raw_pos_args, raw_named_args = cast(_RawArgLists, arg)
                final_pos, final_named = self._finalize_call_args(
                    raw_pos_args, raw_named_args, call_span=self._span_from_meta(meta)
                )
                return ([*final_pos], [*final_named])
        return ([], [])

    def juxt_call_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_suffix: LPAR arg_list? RPAR -> juxt_call_suffix."""
        return ("call", ((), self._juxt_finalized_arg_lists(meta, args)))

    def juxt_typed_call_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_suffix: DCOLON LSQB type_arg_list RSQB LPAR arg_list? RPAR."""
        type_args_val = cast(
            tuple[TypeExpr, ...],
            next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, tuple)
                    and len(arg) > 0
                    and isinstance(arg[0], _ALL_TYPE_EXPRS)
                ),
                (),
            ),
        )
        return ("typed_call", (type_args_val, self._juxt_finalized_arg_lists(meta, args)))

    def type_apply(self, meta: Meta, args: _Args) -> syntax.TypeApply:
        """Apply explicit type arguments to a value without calling it."""
        expr = cast(syntax.Expr, args[0])
        type_args_val = cast(
            tuple[TypeExpr, ...],
            next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, tuple)
                    and len(arg) > 0
                    and isinstance(arg[0], _ALL_TYPE_EXPRS)
                ),
                (),
            ),
        )
        return syntax.TypeApply(
            expr=expr,
            type_args=type_args_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _finalize_call_args(
        self,
        pos_args: list[_RawPosArg],
        named_args: list[_RawNamed],
        *,
        call_span: SourceSpan,
    ) -> tuple[tuple[syntax.Expr, ...], tuple[syntax.NamedArg, ...]]:
        placeholders: list[_RawPlaceholder] = []
        for arg in pos_args:
            if isinstance(arg, _RawPlaceholder):
                placeholders.append(arg)
        for named_arg in named_args:
            if isinstance(named_arg, _RawNamedArg) and isinstance(named_arg.value, _RawPlaceholder):
                placeholders.append(named_arg.value)

        self._validate_placeholders(placeholders, call_span=call_span)

        final_pos: list[syntax.Expr] = []
        for arg in pos_args:
            if isinstance(arg, _RawPlaceholder):
                final_pos.append(self._build_placeholder(arg))
            else:
                final_pos.append(cast(syntax.Expr, arg))

        final_named: list[syntax.NamedArg] = []
        for named_arg in named_args:
            if isinstance(named_arg, syntax.NamedArg):
                final_named.append(named_arg)
            else:
                value = self._build_placeholder(named_arg.value)
                final_named.append(
                    syntax.NamedArg(
                        name=named_arg.name,
                        value=value,
                        span=named_arg.span,
                        node_id=self._next_id(),
                    )
                )
        return (tuple(final_pos), tuple(final_named))

    def _build_placeholder(self, raw: _RawPlaceholder) -> syntax.Placeholder:
        index = int(raw.raw_digits) if raw.raw_digits is not None else None
        return syntax.Placeholder(index=index, span=raw.span, node_id=self._next_id())

    def _validate_placeholders(
        self, placeholders: list[_RawPlaceholder], *, call_span: SourceSpan
    ) -> None:
        if not placeholders:
            return

        bare = [placeholder for placeholder in placeholders if placeholder.raw_digits is None]
        numbered = [
            placeholder for placeholder in placeholders if placeholder.raw_digits is not None
        ]
        for placeholder in numbered:
            assert placeholder.raw_digits is not None
            if placeholder.raw_digits == "0":
                raise AglSyntaxError(
                    "placeholder index must be positive.",
                    span=placeholder.span,
                )
            if placeholder.raw_digits.startswith("0"):
                raise AglSyntaxError(
                    "placeholder index must not have a leading zero.",
                    span=placeholder.span,
                )
        if bare and numbered:
            raise AglSyntaxError(
                "placeholder arguments cannot mix bare and numbered forms in one call.",
                span=numbered[0].span,
            )
        if not numbered:
            return

        seen: dict[int, SourceSpan] = {}
        for placeholder in numbered:
            assert placeholder.raw_digits is not None
            index = int(placeholder.raw_digits)
            if index in seen:
                raise AglSyntaxError(
                    f"placeholder numbered index ?{index} is repeated.",
                    span=placeholder.span,
                )
            seen[index] = placeholder.span
        expected = set(range(1, len(numbered) + 1))
        actual = set(seen)
        if actual != expected:
            missing = sorted(expected - actual)
            detail = f" missing ?{missing[0]}." if missing else ""
            raise AglSyntaxError(
                "numbered placeholder arguments must not have a gap; "
                f"use each index from ?1 to ?{len(numbered)} exactly once.{detail}",
                span=call_span,
            )

    # ------------------------------------------------------------------
    # Call arguments
    # ------------------------------------------------------------------

    def arg_list(self, meta: Meta, args: _Args) -> _RawArgLists:
        """arg_list: arg (COMMA arg)* COMMA?

        Returns (pos_args, named_args) pair for the call builder.
        Duplicate named arg names are rejected with the span of the duplicate.
        Positional args after named args are rejected as a syntax error.
        """
        pos_args: list[_RawPosArg] = []
        named_args: list[_RawNamed] = []
        seen_names: dict[str, SourceSpan] = {}
        seen_named = False
        for a in args:
            if isinstance(a, (syntax.NamedArg, _RawNamedArg)):
                if a.name in seen_names:
                    raise AglSyntaxError(
                        f"duplicate argument {a.name!r}.",
                        span=a.span,
                    )
                seen_names[a.name] = a.span
                named_args.append(a)
                seen_named = True
            elif isinstance(a, _RawPlaceholder) or _is_expr_node(a):
                pos_arg = cast(_RawPosArg, a)
                if seen_named:
                    raise AglSyntaxError(
                        "positional argument after named argument is not allowed.",
                        span=pos_arg.span,
                    )
                pos_args.append(pos_arg)
        return (pos_args, named_args)

    def pos_arg(self, meta: Meta, args: _Args) -> syntax.Expr:
        """pos_arg: expr — transparent wrapper; return the expr."""
        return _find_expr(args)

    def placeholder_arg(self, meta: Meta, args: _Args) -> _RawPlaceholder:
        """placeholder_arg: PLACEHOLDER | PLACEHOLDER_NUM"""
        tok = next(a for a in args if isinstance(a, Token))
        text = str(tok)
        raw_digits = text[1:] if tok.type == "PLACEHOLDER_NUM" else None
        return _RawPlaceholder(raw_digits=raw_digits, span=self._span_from_meta(meta))

    def named_arg(self, meta: Meta, args: _Args) -> syntax.NamedArg | _RawNamedArg:
        """named_arg: field_name EQ named_arg_value"""
        name_tok = _find_name_token(args)
        value = cast(
            syntax.Expr | _RawPlaceholder | _RawInfixChain,
            next(a for a in args[1:] if isinstance(a, _RawPlaceholder) or _is_expr_node(a)),
        )
        span = self._span_from_meta(meta)
        if isinstance(value, _RawPlaceholder):
            return _RawNamedArg(name=str(name_tok), value=value, span=span)
        return syntax.NamedArg(
            name=str(name_tok),
            value=cast(syntax.Expr, value),
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Binary operators
    # ------------------------------------------------------------------

    def _op(
        self, meta: Meta, args: _Args, name: str, builtin: syntax.BinOp | None
    ) -> _InfixOperator:
        return _InfixOperator(name=name, builtin=builtin, span=self._span_from_meta(meta))

    def op_or(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "or", syntax.BinOp.OR)

    def op_and(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "and", syntax.BinOp.AND)

    def op_in(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "in", syntax.BinOp.IN)

    def op_eq(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "==", syntax.BinOp.EQ)

    def op_neq(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "!=", syntax.BinOp.NEQ)

    def op_lt(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "<", syntax.BinOp.LT)

    def op_le(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "<=", syntax.BinOp.LE)

    def op_gt(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, ">", syntax.BinOp.GT)

    def op_ge(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, ">=", syntax.BinOp.GE)

    def op_add(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "+", syntax.BinOp.ADD)

    def op_sub(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "-", syntax.BinOp.SUB)

    def op_mul(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "*", syntax.BinOp.MUL)

    def op_div(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "/", syntax.BinOp.DIV)

    def op_user(self, meta: Meta, args: _Args) -> _InfixOperator:
        tok = next(a for a in args if isinstance(a, Token))
        return self._op(meta, args, str(tok), None)

    def not_prefix(self, meta: Meta, args: _Args) -> object:
        return object()

    def infix_operand(self, meta: Meta, args: _Args) -> _InfixOperand:
        expr = cast(syntax.Expr | _RawInfixChain, next(a for a in args if _is_expr_node(a)))
        not_count = sum(1 for a in args if not isinstance(a, Token) and not _is_expr_node(a))
        return _InfixOperand(
            expr=expr,
            not_count=not_count,
            span=self._span_from_meta(meta),
        )

    def infix_chain(self, meta: Meta, args: _Args) -> syntax.Expr | _RawInfixChain:
        operands = tuple(a for a in args if isinstance(a, _InfixOperand))
        operators = tuple(a for a in args if isinstance(a, _InfixOperator))
        if not operators:
            assert len(operands) == 1
            if operands[0].not_count == 0:
                return operands[0].expr
        return _RawInfixChain(
            operands=operands,
            operators=operators,
            span=self._span_from_meta(meta),
        )

    # ------------------------------------------------------------------
    # Cast operators (as / as?)
    # ------------------------------------------------------------------

    def _make_cast(self, meta: Meta, args: _Args, *, test_only: bool) -> syntax.Cast:
        return syntax.Cast(
            expr=cast(syntax.Expr, args[0]),
            target_type=_find_type_expr(args[1:]),
            test_only=test_only,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def cast_expr(self, meta: Meta, args: _Args) -> syntax.Cast:
        """cast: cast "as" type_expr -> cast_expr"""
        return self._make_cast(meta, args, test_only=False)

    def cast_test(self, meta: Meta, args: _Args) -> syntax.Cast:
        """cast: cast AS_QUESTION type_expr -> cast_test"""
        return self._make_cast(meta, args, test_only=True)

    # ------------------------------------------------------------------
    # Unary operators
    # ------------------------------------------------------------------

    def unary_neg(self, meta: Meta, args: _Args) -> syntax.UnaryNeg:
        operand = cast(syntax.Expr, args[-1])
        return syntax.UnaryNeg(
            operand=operand, span=self._span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # is / is not tests
    # ------------------------------------------------------------------

    def _make_is_test(
        self, meta: Meta, args: _Args, *, qualified: bool, negated: bool
    ) -> syntax.IsTest:
        left = cast(syntax.Expr, args[0])
        name_toks = [a for a in args if _is_name_token(a)]
        module_qual = next((a for a in args if isinstance(a, Qualifier)), None)
        type_qual = next((a for a in args if isinstance(a, TypeQualifier)), None)
        if qualified:
            assert module_qual is not None
            qualifier = type_qual.name if type_qual is not None else None
            variant = str(name_toks[-1])
        else:
            qualifier = None
            variant = str(name_toks[0])
        return syntax.IsTest(
            expr=left,
            qualifier=qualifier,
            variant=variant,
            negated=negated,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=module_qual,
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
            cond=cond,
            body=body,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def if_else_branch(self, meta: Meta, args: _Args) -> syntax.IfBranch:
        """if_else_branch: PIPE? "else" ARROW branch_body"""
        body = _find_expr(args)
        return syntax.IfBranch(
            cond=ELSE,
            body=body,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def if_expr(self, meta: Meta, args: _Args) -> syntax.If:
        branches = tuple(a for a in args if isinstance(a, syntax.IfBranch))
        return syntax.If(
            branches=branches,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: case_expr
    # ------------------------------------------------------------------

    def case_branch(self, meta: Meta, args: _Args) -> syntax.CaseBranch:
        """case_branch: pattern ARROW branch_body"""
        pat = next(a for a in args if isinstance(a, _PATTERN_NODE_TYPES))
        body = _find_expr([a for a in args if not isinstance(a, _PATTERN_NODE_TYPES)])
        assert isinstance(pat, _PATTERN_NODE_TYPES)
        return syntax.CaseBranch(
            pattern=pat,
            body=body,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def first_case_branch(self, meta: Meta, args: _Args) -> syntax.CaseBranch:
        """first_case_branch: PIPE? case_branch"""
        return next(a for a in args if isinstance(a, syntax.CaseBranch))

    def case_branch_seq(self, meta: Meta, args: _Args) -> tuple[syntax.CaseBranch, ...]:
        """case_branch_seq: first_case_branch (PIPE case_branch)*"""
        return tuple(a for a in args if isinstance(a, syntax.CaseBranch))

    def case_body(self, meta: Meta, args: _Args) -> tuple[syntax.CaseBranch, ...]:
        """case_body: case_branch_seq | _INDENT case_branch_seq _NEWLINE? _DEDENT"""
        return _find_case_branch_tuple(args)

    def case_expr(self, meta: Meta, args: _Args) -> syntax.Case:
        """case_expr: "case" or_expr "of" case_body"""
        subject = cast(syntax.Expr, args[0])
        branches = _find_case_branch_tuple(args[1:])
        return syntax.Case(
            subject=subject,
            branches=branches,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: loop_expr
    # ------------------------------------------------------------------

    def break_expr(self, meta: Meta, args: _Args) -> syntax.Break:
        """break_expr: "break" — build a Break AST node."""
        return syntax.Break(
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def continue_expr(self, meta: Meta, args: _Args) -> syntax.Continue:
        """continue_expr: "continue" — build a Continue AST node."""
        return syntax.Continue(
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # range_tail / range_dir / range_by transformers.
    # These produce intermediate tuples consumed by for_clause.

    def range_to(self, meta: Meta, args: _Args) -> bool:
        """range_dir: TO -> range_to — direction flag False (ascending)."""
        return False

    def range_downto(self, meta: Meta, args: _Args) -> bool:
        """range_dir: DOWNTO -> range_downto — direction flag True (descending)."""
        return True

    def range_by(self, meta: Meta, args: _Args) -> syntax.Expr:
        """range_by: BY or_expr — return the step expression."""
        return cast(syntax.Expr, _find_non_token(args))

    def range_tail(self, meta: Meta, args: _Args) -> tuple[bool, syntax.Expr, syntax.Expr | None]:
        """range_tail: range_dir or_expr range_by?

        Returns (is_downto, to_bound_expr, by_step_expr_or_None).
        The grammar guarantees exactly: one bool (range_dir result) followed by
        one or two Exprs (bound from or_expr, then optional step from range_by).
        """
        # Separate the direction flag from the expression arguments.
        is_down: bool = next(a for a in args if isinstance(a, bool))
        exprs = [cast(syntax.Expr, a) for a in args if _is_expr_node(a)]
        assert len(exprs) in (1, 2), f"range_tail: unexpected expr count {len(exprs)}"
        to_bound = exprs[0]
        by_step: syntax.Expr | None = exprs[1] if len(exprs) == 2 else None
        return (is_down, to_bound, by_step)

    def for_clause(
        self, meta: Meta, args: _Args
    ) -> tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None]:
        """for_clause: "for" name "in" or_expr range_tail? _NEWLINE?

        Returns a 5-tuple:
          (var_name, start_expr, range_to_expr, range_down, range_by_expr)

        For a collection for (no range_tail):
          range_to_expr=None, range_down=False, range_by_expr=None.
        For a range for (range_tail present):
          range_to_expr is the upper/lower bound; range_down is True for downto;
          range_by_expr is the step or None for default step.
        """
        name_tok = next(a for a in args if isinstance(a, Token))
        # Separate range_tail tuple (3-element tuple starting with bool) from
        # the or_expr (the start/collection expression).
        range_tail_result = next(
            (
                cast(tuple[bool, syntax.Expr, syntax.Expr | None], a)
                for a in args
                if isinstance(a, tuple) and len(a) == 3 and isinstance(a[0], bool)
            ),
            None,
        )
        start_expr = cast(syntax.Expr, next(a for a in args if _is_expr_node(a)))
        if range_tail_result is not None:
            range_down, range_to, range_by = range_tail_result
        else:
            range_down, range_to, range_by = False, None, None
        return (str(name_tok), start_expr, range_to, range_down, range_by)

    def while_clause(self, meta: Meta, args: _Args) -> syntax.Expr:
        """while_clause: "while" or_expr _NEWLINE?

        Returns the condition expression.
        """
        return cast(syntax.Expr, _find_non_token(args))

    # Type alias for the extended for_clause result tuple.
    _ForClauseResult = tuple[str, syntax.Expr, "syntax.Expr | None", bool, "syntax.Expr | None"]

    def loop_clauses(
        self,
        meta: Meta,
        args: _Args,
    ) -> tuple[
        tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None] | None,
        syntax.Expr | None,
    ]:
        """loop_clauses: for_clause? while_clause?

        Returns a 2-tuple (for_result_or_None, while_result_or_None).
        Detection is type-based: for_clause returns a 5-tuple starting with a str,
        while_clause returns an Expr directly.
        """
        # for_clause returns a 5-tuple (str, Expr, Expr|None, bool, Expr|None).
        # while_clause returns an Expr.  Distinguish by the 5-tuple signature.
        for_result: tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None] | None = (
            next(
                (
                    cast("tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None]", a)
                    for a in args
                    if isinstance(a, tuple) and len(a) == 5 and isinstance(a[0], str)
                ),
                None,
            )
        )
        while_result: syntax.Expr | None = next(
            (cast(syntax.Expr, a) for a in args if _is_expr_node(a)),
            None,
        )
        return (for_result, while_result)

    def loop_bound(self, meta: Meta, args: _Args) -> syntax.Expr:
        """loop_bound: DO_LSQB or_expr RSQB — return the bound expression.

        The value is validated (must be a non-negative ``int``) at runtime, not
        here: the bound is an arbitrary expression and its value is unknown
        until the loop is reached.
        """
        return cast(syntax.Expr, _find_non_token(args))

    def do_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """do_body: suite_expr | inline_seq — pass through the inner expr."""
        inner = _find_non_token(args)
        return cast(syntax.Expr, inner)

    def inline_seq(self, meta: Meta, args: _Args) -> syntax.Block:
        """inline_seq: inline_item (SEMICOLON inline_item)*

        Build a Block from the inline items (or_exprs and binders).
        """
        items = tuple(
            cast(syntax.Item, a) for a in args if a is not None and not isinstance(a, Token)
        )
        return syntax.Block(
            items=items,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def inline_item(self, meta: Meta, args: _Args) -> syntax.Item:
        """inline_item: binder | or_expr — transparent."""
        inner = _find_non_token(args)
        return cast(syntax.Item, inner)

    def loop_until(self, meta: Meta, args: _Args) -> syntax.Expr:
        """loop_until: "until" or_expr — return the condition expression."""
        return cast(syntax.Expr, _find_non_token(args))

    def loop_done(self, meta: Meta, args: _Args) -> None:
        """loop_done: "done" — return None sentinel meaning 'done' (≡ until false)."""
        return None

    def loop_expr(self, meta: Meta, args: _Args) -> syntax.Loop:
        """loop_expr: loop_clauses "do" loop_bound? do_body loop_end

        In LALR mode, absent optional rules are simply not included in the
        tree (maybe_placeholders does not insert None).  Non-Token children:
          0    : loop_clauses result  — always present (tuple)
          1    : loop_bound result    — present only when loop_bound is given
          last : loop_end result      — always present last (Expr or None)
          last-1: do_body result      — always present second-to-last (Expr)

        String terminal ``"do"`` is stripped by Lark and never appears.
        """
        # String terminals ("do") are stripped by Lark; loop_end returns Expr|None
        # (loop_until → Expr, loop_done → None) — neither is a Token, so don't
        # filter on isinstance(Token).  Only the terminal tokens (DO_LSQB, RSQB
        # etc.) inside sub-rules are filtered by *those* rules' transformers.
        children = [a for a in args if not isinstance(a, Token)]
        # Invariant: 3 or 4 children (loop_bound? is the variable one).
        assert len(children) in (3, 4), f"loop_expr: unexpected children count {len(children)}"

        clauses = cast(
            "tuple["
            "  tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None] | None,"
            "  syntax.Expr | None"
            "]",
            children[0],
        )
        for_var: str | None = None
        for_iter: syntax.Expr | None = None
        for_range_to: syntax.Expr | None = None
        for_range_down: bool = False
        for_range_by: syntax.Expr | None = None
        if clauses[0] is not None:
            for_var, for_iter, for_range_to, for_range_down, for_range_by = clauses[0]
        while_cond: syntax.Expr | None = clauses[1]

        # Invariants: range for requires var + start + bound; collection for has no bound.
        if for_range_to is not None:
            assert for_var is not None, "loop_expr: range for missing var"
            assert for_iter is not None, "loop_expr: range for missing start expression"
        else:
            assert not for_range_down, "loop_expr: range_down set without range_to"
            assert for_range_by is None, "loop_expr: range_by set without range_to"

        # loop_end is always the last child; do_body is second-to-last.
        until_cond: syntax.Expr | None = cast("syntax.Expr | None", children[-1])
        body = cast(syntax.Expr, children[-2])
        # loop_bound is present only when len == 4.
        bound: syntax.Expr | None = (
            cast("syntax.Expr | None", children[1]) if len(children) == 4 else None
        )
        # A non-positive bound (e.g. do[0] or do[-1]) is NOT rejected here: per
        # the loop design a bound n <= 0 runs the body zero times and
        # completes normally.  The lowerer's runtime bound check handles it.

        return syntax.Loop(
            for_var=for_var,
            for_iter=for_iter,
            for_range_to=for_range_to,
            for_range_down=for_range_down,
            for_range_by=for_range_by,
            while_cond=while_cond,
            bound=bound,
            body=body,
            until_cond=until_cond,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
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
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def catch_pattern(self, meta: Meta, args: _Args) -> tuple[str | None, str | None]:
        """catch_pattern: name ("as" name)?

        Handles any NAME. Wildcard is "_" (NAME).
        """
        name_toks = [a for a in args if _is_name_token(a)]
        if not name_toks:  # pragma: no cover
            return (None, None)
        first_name = str(name_toks[0])
        binding: str | None = str(name_toks[1]) if len(name_toks) >= 2 else None
        if first_name == "_":
            exc_type: str | None = None
        else:
            exc_type = first_name
        return (exc_type, binding)

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
                if (
                    len(a) == 2
                    and (a[0] is None or isinstance(a[0], str))
                    and (a[1] is None or isinstance(a[1], str))
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
            exc_type=exc_type,
            binding=binding,
            body=body,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def try_expr(self, meta: Meta, args: _Args) -> syntax.Try:
        """try_expr: "try" try_body (catch_clause)+"""
        handlers = [a for a in args if isinstance(a, syntax.CatchClause)]
        try_body = next(
            a
            for a in args
            if a is not None and not isinstance(a, Token) and not isinstance(a, syntax.CatchClause)
        )
        return syntax.Try(
            body=cast(syntax.Expr, try_body),
            handlers=tuple(handlers),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Control flow: raise_expr
    # ------------------------------------------------------------------

    def raise_expr(self, meta: Meta, args: _Args) -> syntax.Raise:
        exc = _find_expr(args)
        return syntax.Raise(exc=exc, span=self._span_from_meta(meta), node_id=self._next_id())

    def return_expr(self, meta: Meta, args: _Args) -> syntax.Return:
        value = next((cast(syntax.Expr, a) for a in args if _is_expr_node(a)), None)
        return syntax.Return(value=value, span=self._span_from_meta(meta), node_id=self._next_id())

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
        """name → WildcardPattern (when value is "_") or VarPattern."""
        tok = args[0]
        assert isinstance(tok, Token)
        if str(tok) == "_":
            return syntax.WildcardPattern(span=self._span_from_meta(meta), node_id=self._next_id())
        return syntax.VarPattern(
            name=str(tok), span=self._span_from_meta(meta), node_id=self._next_id()
        )

    def pat_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        """pat_constructor: name LPAR pattern_fields? RPAR"""
        name_toks = [a for a in args if _is_name_token(a)]
        assert len(name_toks) >= 1, "pat_constructor: expected name token"
        name = str(name_toks[0])
        positional: tuple[syntax.Pattern, ...] = ()
        named: tuple[syntax.PatternField, ...] = ()
        for a in args:
            if isinstance(a, _PatternFieldsSplit):
                positional = a.positional
                named = a.named
        return syntax.ConstructorPattern(
            qualifier=None,
            name=name,
            positional=positional,
            named=named,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _literal_pattern(
        self,
        literal: syntax.IntLit
        | syntax.DecimalLit
        | syntax.BoolLit
        | syntax.StringLit
        | syntax.NullLit,
        meta: Meta,
    ) -> syntax.LiteralPattern:
        return syntax.LiteralPattern(
            literal=literal, span=self._span_from_meta(meta), node_id=self._next_id()
        )

    def pat_lit_int(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        lit = syntax.IntLit(
            value=int(str(tok)), span=self._span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_decimal(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        lit = syntax.DecimalLit(
            value=decimal.Decimal(str(tok)),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_true(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(value=True, span=self._span_from_meta(meta), node_id=self._next_id())
        return self._literal_pattern(lit, meta)

    def pat_lit_false(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(value=False, span=self._span_from_meta(meta), node_id=self._next_id())
        return self._literal_pattern(lit, meta)

    def pat_lit_null(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.NullLit(span=self._span_from_meta(meta), node_id=self._next_id())
        return self._literal_pattern(lit, meta)

    def pat_lit_str(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tmpl = _require_literal_string(
            args[0], "Pattern string literals cannot contain interpolation."
        )
        return self._literal_pattern(tmpl, meta)

    def pattern_fields(self, meta: Meta, args: _Args) -> _PatternFieldsSplit:
        """Collect pattern_field children into a split of positional and named."""
        positional: list[syntax.Pattern] = []
        named: list[syntax.PatternField] = []
        seen_named = False
        for a in args:
            if isinstance(a, syntax.PatternField):
                seen_named = True
                named.append(a)
            elif isinstance(a, _PATTERN_NODE_TYPES):
                if seen_named:
                    raise AglSyntaxError(
                        "Positional sub-pattern after a named field pattern.",
                        span=_span_from_meta(meta),
                    )
                positional.append(a)
        return _PatternFieldsSplit(positional=tuple(positional), named=tuple(named))

    def pat_field_named(self, meta: Meta, args: _Args) -> syntax.PatternField:
        """pat_field_named: name EQ pattern"""
        name_tok = _find_name_token(args)
        pat = next((a for a in args if isinstance(a, _PATTERN_NODE_TYPES)), None)
        assert pat is not None
        assert isinstance(pat, _PATTERN_NODE_TYPES)
        return syntax.PatternField(
            name=str(name_tok),
            pattern=pat,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def pat_field_positional(self, meta: Meta, args: _Args) -> syntax.Pattern:
        """pat_field_positional: pattern — return the sub-pattern directly."""
        pat = next((a for a in args if isinstance(a, _PATTERN_NODE_TYPES)), None)
        assert pat is not None
        assert isinstance(pat, _PATTERN_NODE_TYPES)
        return pat

    # ------------------------------------------------------------------
    # Import declaration
    # ------------------------------------------------------------------

    def _import_decl_from_args(
        self,
        meta: Meta,
        args: _Args,
        *,
        wildcard: bool,
    ) -> syntax.ImportDecl:
        """Shared builder for import_decl_plain and import_decl_wildcard."""
        module_path: tuple[str, ...] = ()
        is_open = False
        alias: str | None = None
        mode = ImportMode.ALL
        items: tuple[syntax.ImportItem, ...] = ()

        for a in args:
            if isinstance(a, Token) and a.type == "OPEN":
                is_open = True
            elif isinstance(a, Token) and a.type == "MODPATH":
                module_path = tuple(str(a).split("/"))
            elif isinstance(a, Token):
                # Skip IMPORT, SLASH, STAR, etc.
                pass
            elif type(a) is str:
                # import_alias result: plain str (not Token, which is also a str subclass)
                alias = a
            else:
                mode, items = cast(tuple[ImportMode, tuple[syntax.ImportItem, ...]], a)

        if is_open and mode is ImportMode.USING:
            raise AglSyntaxError(
                "`open import` cannot use a `using` clause.", span=self._span_from_meta(meta)
            )

        return syntax.ImportDecl(
            module_path=module_path,
            wildcard=wildcard,
            is_open=is_open,
            alias=alias,
            mode=mode,
            items=items,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def import_decl_plain(self, meta: Meta, args: _Args) -> syntax.ImportDecl:
        """import_decl_plain: OPEN? IMPORT MODPATH import_alias? import_clause?"""
        return self._import_decl_from_args(meta, args, wildcard=False)

    def import_decl_wildcard(self, meta: Meta, args: _Args) -> syntax.ImportDecl:
        """import_decl_wildcard: OPEN? IMPORT MODPATH SLASH STAR import_alias? import_clause?"""
        return self._import_decl_from_args(meta, args, wildcard=True)

    def import_alias(self, meta: Meta, args: _Args) -> str:
        """import_alias: "as" name — return the alias name as a plain str."""
        tok = next(a for a in args if _is_name_token(a))
        return str(tok)

    def _hiding_items(self, meta: Meta, args: _Args) -> tuple[syntax.ImportItem, ...]:
        """Build hiding ImportItem tuples from the names in a HIDING clause."""
        hiding_names = [str(a) for a in args if _is_name_token(a)]
        return tuple(
            syntax.ImportItem(
                name=name,
                rename=None,
                span=self._span_from_meta(meta),
                node_id=self._next_id(),
            )
            for name in hiding_names
        )

    def import_clause_using(
        self, meta: Meta, args: _Args
    ) -> tuple[ImportMode, tuple[syntax.ImportItem, ...]]:
        """import_clause_using: USING import_item (COMMA import_item)*"""
        import_items = tuple(a for a in args if isinstance(a, syntax.ImportItem))
        return (ImportMode.USING, import_items)

    def import_clause_hiding(
        self, meta: Meta, args: _Args
    ) -> tuple[ImportMode, tuple[syntax.ImportItem, ...]]:
        """import_clause_hiding: HIDING name (COMMA name)*"""
        return (ImportMode.HIDING, self._hiding_items(meta, args))

    def import_item_rename(self, meta: Meta, args: _Args) -> syntax.ImportItem:
        """import_item_rename: name "as" name"""
        name_toks = [str(a) for a in args if _is_name_token(a)]
        return syntax.ImportItem(
            name=name_toks[0],
            rename=name_toks[1],
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def import_item_plain(self, meta: Meta, args: _Args) -> syntax.ImportItem:
        """import_item_plain: name"""
        name_toks = [str(a) for a in args if _is_name_token(a)]
        return syntax.ImportItem(
            name=name_toks[0],
            rename=None,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Export declaration
    # ------------------------------------------------------------------

    def _export_decl_from_args(
        self,
        meta: Meta,
        args: _Args,
        *,
        wildcard: bool,
    ) -> syntax.ExportDecl:
        """Shared builder for export_decl_plain and export_decl_wildcard."""
        module_path: tuple[str, ...] = ()
        mode = ImportMode.ALL
        items: tuple[syntax.ExportItem, ...] = ()

        for a in args:
            if isinstance(a, Token) and a.type == "MODPATH":
                module_path = tuple(str(a).split("/"))
            elif isinstance(a, Token):
                # Skip EXPORT, SLASH, STAR, etc.
                pass
            else:
                mode, items = cast(tuple[ImportMode, tuple[syntax.ExportItem, ...]], a)

        return syntax.ExportDecl(
            module_path=module_path,
            wildcard=wildcard,
            mode=mode,
            items=items,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def export_decl_plain(self, meta: Meta, args: _Args) -> syntax.ExportDecl:
        """export_decl_plain: EXPORT MODPATH export_clause?"""
        return self._export_decl_from_args(meta, args, wildcard=False)

    def export_decl_wildcard(self, meta: Meta, args: _Args) -> syntax.ExportDecl:
        """export_decl_wildcard: EXPORT MODPATH SLASH STAR export_clause?"""
        return self._export_decl_from_args(meta, args, wildcard=True)

    def _export_hiding_items(self, meta: Meta, args: _Args) -> tuple[syntax.ExportItem, ...]:
        """Build hiding ExportItem tuples from the names in a HIDING clause."""
        hiding_names = [str(a) for a in args if _is_name_token(a)]
        return tuple(
            syntax.ExportItem(
                name=name,
                rename=None,
                span=self._span_from_meta(meta),
                node_id=self._next_id(),
            )
            for name in hiding_names
        )

    def export_clause_using(
        self, meta: Meta, args: _Args
    ) -> tuple[ImportMode, tuple[syntax.ExportItem, ...]]:
        """export_clause_using: USING export_item (COMMA export_item)*"""
        export_items = tuple(a for a in args if isinstance(a, syntax.ExportItem))
        return (ImportMode.USING, export_items)

    def export_clause_hiding(
        self, meta: Meta, args: _Args
    ) -> tuple[ImportMode, tuple[syntax.ExportItem, ...]]:
        """export_clause_hiding: HIDING name (COMMA name)*"""
        return (ImportMode.HIDING, self._export_hiding_items(meta, args))

    def export_item_rename(self, meta: Meta, args: _Args) -> syntax.ExportItem:
        """export_item_rename: name "as" name"""
        name_toks = [str(a) for a in args if _is_name_token(a)]
        return syntax.ExportItem(
            name=name_toks[0],
            rename=name_toks[1],
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def export_item_plain(self, meta: Meta, args: _Args) -> syntax.ExportItem:
        """export_item_plain: name"""
        name_toks = [str(a) for a in args if _is_name_token(a)]
        return syntax.ExportItem(
            name=name_toks[0],
            rename=None,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Private declarations
    # ------------------------------------------------------------------

    def private_record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        """private_record_def: PRIVATE record_def"""
        rec = next(a for a in args if isinstance(a, syntax.RecordDef))
        return syntax.RecordDef(
            name=rec.name,
            fields=rec.fields,
            type_params=rec.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=True,
            is_builtin=rec.is_builtin,
        )

    def private_enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        """private_enum_def: PRIVATE enum_def"""
        e = next(a for a in args if isinstance(a, syntax.EnumDef))
        return syntax.EnumDef(
            name=e.name,
            variants=e.variants,
            type_params=e.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=True,
            is_builtin=e.is_builtin,
        )

    def private_exception_def(self, meta: Meta, args: _Args) -> syntax.ExceptionDef:
        """private_exception_def: PRIVATE exception_def"""
        exc = next(a for a in args if isinstance(a, syntax.ExceptionDef))
        return syntax.ExceptionDef(
            name=exc.name,
            fields=exc.fields,
            base=exc.base,
            type_params=exc.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=True,
            is_builtin=exc.is_builtin,
        )

    def private_type_alias(self, meta: Meta, args: _Args) -> syntax.TypeAlias:
        """private_type_alias: PRIVATE type_alias"""
        ta = next(a for a in args if isinstance(a, syntax.TypeAlias))
        return syntax.TypeAlias(
            name=ta.name,
            type_expr=ta.type_expr,
            type_params=ta.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=True,
        )

    def _private_wrap_func_def(self, meta: Meta, f: syntax.FuncDef) -> syntax.FuncDef:
        """Shared construction for ``private`` wrapping of a ``FuncDef``.

        Shared by ``private_func_def`` and ``private_extern_func_def`` — both
        re-emit a copy of the wrapped ``FuncDef`` with ``is_private=True``,
        preserving ``is_builtin``/``is_extern``.
        """
        return syntax.FuncDef(
            name=f.name,
            params=f.params,
            return_type=f.return_type,
            body=f.body,
            type_params=f.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=True,
            is_builtin=f.is_builtin,
            is_extern=f.is_extern,
        )

    def private_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """private_func_def: PRIVATE func_def"""
        f = next(a for a in args if isinstance(a, syntax.FuncDef))
        return self._private_wrap_func_def(meta, f)

    def private_extern_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """private_extern_func_def: PRIVATE extern_func_def"""
        f = next(a for a in args if isinstance(a, syntax.FuncDef))
        return self._private_wrap_func_def(meta, f)

    def builtin_record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        """builtin_record_def: BUILTIN record_def"""
        rec = next(a for a in args if isinstance(a, syntax.RecordDef))
        return syntax.RecordDef(
            name=rec.name,
            fields=rec.fields,
            type_params=rec.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=rec.is_private,
            is_builtin=True,
        )

    def builtin_enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        """builtin_enum_def: BUILTIN enum_def"""
        e = next(a for a in args if isinstance(a, syntax.EnumDef))
        return syntax.EnumDef(
            name=e.name,
            variants=e.variants,
            type_params=e.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=e.is_private,
            is_builtin=True,
        )

    def builtin_exception_def(self, meta: Meta, args: _Args) -> syntax.ExceptionDef:
        """builtin_exception_def: BUILTIN exception_def"""
        exc = next(a for a in args if isinstance(a, syntax.ExceptionDef))
        return syntax.ExceptionDef(
            name=exc.name,
            fields=exc.fields,
            base=exc.base,
            type_params=exc.type_params,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_private=exc.is_private,
            is_builtin=True,
        )

    # ------------------------------------------------------------------
    # Qualified refs
    # ------------------------------------------------------------------

    def qual_prefix(self, meta: Meta, args: _Args) -> Qualifier:
        """qual_prefix: MODQUAL | DCOLON"""
        tok = args[0]
        assert isinstance(tok, Token)
        if tok.type == "MODQUAL":
            spelling = str(tok)
            anchored = spelling.startswith("/")
            segments = tuple(spelling.removeprefix("/").split("/"))
        else:
            # DCOLON — self-reference, empty segments
            anchored = False
            segments = ()
        return Qualifier(
            segments=segments,
            span=self._span_from_token(tok),
            node_id=self._next_id(),
            anchored=anchored,
        )

    def type_qual(self, meta: Meta, args: _Args) -> TypeQualifier:
        """type_qual: MODQUAL."""
        tok = args[0]
        assert isinstance(tok, Token)
        spelling = str(tok)
        anchored = spelling.startswith("/")
        segments = tuple(spelling.removeprefix("/").split("/"))
        if anchored or len(segments) != 1:
            raise AglSyntaxError(
                "A type qualifier after '::' must be a single type name.",
                span=self._span_from_token(tok),
            )
        return TypeQualifier(
            name=segments[0],
            type_args=None,
            span=self._span_from_token(tok),
            node_id=self._next_id(),
        )

    def qual_var_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        """qual_var_ref: qual_prefix type_qual? NAME"""
        qual = next(a for a in args if isinstance(a, Qualifier))
        type_qual = next((a for a in args if isinstance(a, TypeQualifier)), None)
        name_tok = next(a for a in args if _is_name_token(a))
        return syntax.VarRef(
            name=str(name_tok),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=qual,
            type_qualifier=type_qual,
        )

    def qual_named_type(self, meta: Meta, args: _Args) -> NameT:
        """qual_prefix NAME in type position."""
        qual = next(a for a in args if isinstance(a, Qualifier))
        name_tok = next(a for a in args if _is_name_token(a))
        return NameT(
            name=str(name_tok),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=qual,
        )

    def applied_qual_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        """Type-qualified constructor with explicit type arguments."""
        qual = next((a for a in args if isinstance(a, Qualifier)), None)
        name_toks = [a for a in args if _is_name_token(a)]
        type_name = str(name_toks[0])
        variant_name = str(name_toks[-1])
        type_args_val = cast(
            tuple[TypeExpr, ...],
            next(
                (
                    arg
                    for arg in args
                    if isinstance(arg, tuple)
                    and len(arg) > 0
                    and isinstance(arg[0], _ALL_TYPE_EXPRS)
                ),
                (),
            ),
        )
        type_qual = TypeQualifier(
            name=type_name,
            type_args=type_args_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )
        return syntax.VarRef(
            name=variant_name,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=qual,
            type_qualifier=type_qual,
        )

    def pat_qual_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        """pat_qual_constructor: qual_prefix type_qual? NAME (LPAR pattern_fields? RPAR)?"""
        qual = next(a for a in args if isinstance(a, Qualifier))
        type_qual = next((a for a in args if isinstance(a, TypeQualifier)), None)
        name_tok = next(a for a in args if _is_name_token(a))
        positional: tuple[syntax.Pattern, ...] = ()
        named: tuple[syntax.PatternField, ...] = ()
        for a in args:
            if isinstance(a, _PatternFieldsSplit):
                positional = a.positional
                named = a.named
        return syntax.ConstructorPattern(
            qualifier=type_qual.name if type_qual is not None else None,
            name=str(name_tok),
            positional=positional,
            named=named,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            module_qualifier=qual,
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
        span = self._span_from_meta(meta)
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
            span=self._span_from_meta(meta),
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
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # List and dict literals
    # ------------------------------------------------------------------

    def lit_list(self, meta: Meta, args: _Args) -> syntax.ListLit:
        """lit_list: LSQB (expr (COMMA expr)* COMMA?)? RSQB"""
        elements = tuple(
            cast(syntax.Expr, a) for a in args if a is not None and not isinstance(a, Token)
        )
        return syntax.ListLit(
            elements=elements,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_dict(self, meta: Meta, args: _Args) -> syntax.DictLit:
        """lit_dict: LBRACE (dict_entry (COMMA dict_entry)* COMMA?)? RBRACE"""
        entries = tuple(a for a in args if isinstance(a, syntax.DictEntry))
        return syntax.DictLit(
            entries=entries,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_str(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: template COLON expr — quoted string key."""
        non_tokens = [a for a in args if a is not None and not isinstance(a, Token)]
        assert len(non_tokens) >= 2, f"dict_entry_str: expected key + expr, got {args!r}"
        key_lit = _require_literal_string(
            non_tokens[0],
            "dict keys must be literal strings (no interpolation).",
        )
        val_expr = cast(syntax.Expr, non_tokens[1])
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_name(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: field_name COLON expr — identifier shorthand key."""
        name_tok = _find_name_token(args)
        val_expr = _find_expr(args[1:])
        key_lit = syntax.StringLit(
            value=str(name_tok),
            span=self._span_from_token(name_tok),
            node_id=self._next_id(),
        )
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=self._span_from_meta(meta),
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

    ``name`` matches ``NAME`` or ``OP_NAME``. ``field_name`` also admits a few
    keyword tokens: ``AGENT``, ``TO``, ``DOWNTO``, ``BY``. All arrive here as
    plain Tokens; callers treat ``str(token)`` as the name string.
    """
    for a in args:
        if _is_name_token(a) or (
            isinstance(a, Token) and a.type in ("AGENT", "TO", "DOWNTO", "BY")
        ):
            return a
    raise AssertionError(f"_find_name_token: no name token found in {args!r}")  # pragma: no cover


def _is_name_token(value: object) -> TypeGuard[Token]:
    return isinstance(value, Token) and value.type in _NAME_TOKEN_TYPES


def _is_field_tuple(a: object) -> bool:
    """True iff *a* is a field-list result (``tuple`` of ``Param | _ParamMarker``).

    An empty tuple is treated as a field tuple (the ``field_list?`` absent case).
    Markers may appear at position 0 in the raw entries returned by ``field_list``
    before zone resolution, so ``_ParamMarker`` is accepted here too.
    """
    return isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], (syntax.Param, _ParamMarker)))


def _find_field_tuple(args: _Args) -> tuple[syntax.Param, ...]:
    result = next((a for a in args if _is_field_tuple(a)), None)
    if result is None:  # pragma: no cover
        raise AssertionError(f"_find_field_tuple: no field tuple found in {args!r}")
    return cast(tuple[syntax.Param, ...], result)


def _resolve_params(
    entries: _RawEntries,
    *,
    default_kind: syntax.ParamKind,
) -> tuple[syntax.Param, ...]:
    """Resolve a marker/param interleaving to ``Param``s with concrete ``kind``s.

    Algorithm:

    - **No marker** → every ``Param`` gets ``default_kind``.
    - **≥1 marker** (pure positional reading; ``default_kind`` is ignored):
      - Markers must be strictly increasing by zone (rejects duplicates and
        out-of-order); ``@pos`` must be the first entry (no ``Param`` before it).
      - Zone for params *before* the first marker = one zone below the first
        marker's zone.
      - Walk left-to-right: marker → set ``current``; param → assign ``current``.
    """
    markers = [e for e in entries if isinstance(e, _ParamMarker)]
    if not markers:
        # No marker: apply the per-context default to every param.
        return tuple(replace(p, kind=default_kind) for p in entries if isinstance(p, syntax.Param))

    # Validate: markers must be strictly increasing by zone.
    last_order = -1
    for m in markers:
        order = _ZONE_ORDER[m.zone]
        if order <= last_order:
            raise AglSyntaxError(
                f"duplicate / out-of-order parameter marker {m.label!r}.",
                span=m.span,
            )
        last_order = order

    # Validate: @pos must be leading (no Param may precede it in entries).
    pos_marker = next((m for m in markers if m.zone == syntax.ParamKind.POSITIONAL_ONLY), None)
    if pos_marker is not None:
        # Find the index of pos_marker in entries (by identity).
        pos_idx = next(i for i, e in enumerate(entries) if e is pos_marker)
        if any(isinstance(e, syntax.Param) for e in entries[:pos_idx]):
            raise AglSyntaxError(
                f"positional-only marker {pos_marker.label!r} must lead the parameter list.",
                span=pos_marker.span,
            )

    # Determine zone for params that appear before the first marker.
    first_marker = markers[0]
    first_order = _ZONE_ORDER[first_marker.zone]
    # initial_kind is None only when @pos is first (no params allowed before it).
    initial_kind: syntax.ParamKind | None = (
        None if first_order == 0 else _ZONE_BY_ORDER[first_order - 1]
    )

    current = initial_kind
    result: list[syntax.Param] = []
    for e in entries:
        if isinstance(e, _ParamMarker):
            current = e.zone
        else:
            assert current is not None  # guaranteed: @pos check above
            result.append(replace(e, kind=current))

    return tuple(result)


def _is_variant_tuple(a: object) -> bool:
    return isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], syntax.VariantDef))


def _find_variant_tuple(args: _Args) -> tuple[syntax.VariantDef, ...]:
    result = next((a for a in args if _is_variant_tuple(a)), None)
    if result is None:  # pragma: no cover
        raise AssertionError(f"_find_variant_tuple: no variant tuple found in {args!r}")
    return cast(tuple[syntax.VariantDef, ...], result)


def _is_case_branch_tuple(a: object) -> bool:
    return isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], syntax.CaseBranch))


def _find_case_branch_tuple(args: _Args) -> tuple[syntax.CaseBranch, ...]:
    result = next((a for a in args if _is_case_branch_tuple(a)), None)
    if result is None:  # pragma: no cover
        raise AssertionError(f"_find_case_branch_tuple: no case branch tuple found in {args!r}")
    return cast(tuple[syntax.CaseBranch, ...], result)


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
    return isinstance(a, syntax.Expr) or isinstance(a, _RawInfixChain)


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


def resolve_infix_fixity(
    decls: "Iterable[syntax.InfixDecl]",
    ambient: "Mapping[str, tuple[int, syntax.InfixAssoc]] | None" = None,
) -> dict[str, tuple[int, syntax.InfixAssoc]]:
    """Resolve an ordered sequence of infix declarations into a fixity table.

    Returns a mapping of user operator name → ``(priority, associativity)``.
    *ambient* is an already-resolved fixity table carried over from a prior
    context (e.g. earlier REPL entries); its entries may be overridden by a
    later redeclaration in *decls*, mirroring how ``let``/``record``
    redefinitions shadow in the REPL.  Relative priorities (``at prio OP ±``
    *n*) resolve against built-in operators, the ambient table, and any earlier
    declaration in *decls*.

    Validates: a user operator may not redeclare a built-in, and the same name
    may not be declared twice within *decls* (redeclaration across the ambient
    boundary is allowed and overrides).  This is the single source of truth for
    infix-priority resolution, shared by the parser and the REPL session.
    """
    # Built-in operators are read-only inputs for relative-priority resolution
    # (they can never be user-redeclared, so they are never emitted in the result).
    resolved: dict[str, tuple[int, syntax.InfixAssoc]] = dict(ambient) if ambient else {}
    seen_user: set[str] = set()
    for decl in decls:
        if decl.name in _BUILTIN_INFIX_PRIORITIES:
            raise AglSyntaxError(
                f"Cannot redeclare built-in operator '{decl.name}' as a user infix operator.",
                span=decl.span,
            )
        if decl.name in seen_user:
            raise AglSyntaxError(
                f"Infix operator '{decl.name}' is already declared.",
                span=decl.span,
            )
        seen_user.add(decl.name)
        priority = _resolve_infix_priority(decl, resolved)
        resolved[decl.name] = (priority, decl.assoc)
    return resolved


def _resolve_infix_priority(
    decl: syntax.InfixDecl,
    resolved: "Mapping[str, tuple[int, syntax.InfixAssoc]]",
) -> int:
    """Resolve a single declaration's priority against built-ins + *resolved*."""
    if decl.priority is not None:
        return decl.priority
    if decl.priority_base is not None:
        base = resolved.get(decl.priority_base)
        if base is None:
            base_priority = _BUILTIN_INFIX_PRIORITIES.get(decl.priority_base)
            if base_priority is None:
                raise AglSyntaxError(
                    f"Unknown operator '{decl.priority_base}' in priority reference.",
                    span=decl.span,
                )
            base = (base_priority, _BUILTIN_INFIX_ASSOC[decl.priority_base])
        return base[0] + decl.priority_delta
    return _DEFAULT_USER_INFIX_PRIORITY


def _operator_table_from_decls(
    items: tuple[syntax.Item, ...],
    ambient: "Mapping[str, tuple[int, syntax.InfixAssoc]] | None" = None,
) -> dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]]:
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]] = {
        name: (
            priority,
            _BUILTIN_INFIX_ASSOC[name],
            _BUILTIN_INFIX_OPS[name],
        )
        for name, priority in _BUILTIN_INFIX_PRIORITIES.items()
    }
    user_decls = [item for item in items if isinstance(item, syntax.InfixDecl)]
    for name, (priority, assoc) in resolve_infix_fixity(user_decls, ambient).items():
        table[name] = (priority, assoc, None)
    return table


def _rewrite_block_infix(
    block: syntax.Block,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.Block:
    rewritten = tuple(_rewrite_item(item, table, builder) for item in block.items)
    return replace(block, items=rewritten)


def _rewrite_item(
    item: _RawItem,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.Item:
    if isinstance(item, _RawInfixChain):
        return _resolve_infix_chain(item, table, builder)
    if isinstance(item, syntax.Expr):
        return _rewrite_expr(item, table, builder)
    if isinstance(item, syntax.LetDecl):
        return replace(item, value=_rewrite_expr(item.value, table, builder))
    if isinstance(item, syntax.VarDecl):
        return replace(item, value=_rewrite_expr(item.value, table, builder))
    if isinstance(item, syntax.AssignStmt):
        return replace(
            item,
            target=_rewrite_assign_target(item.target, table, builder),
            value=_rewrite_expr(item.value, table, builder),
        )
    if isinstance(item, syntax.FuncDef):
        return replace(
            item,
            params=tuple(_rewrite_param(p, table, builder) for p in item.params),
            body=None if item.body is None else _rewrite_expr(item.body, table, builder),
        )
    if isinstance(item, syntax.RecordDef):
        return replace(item, fields=tuple(_rewrite_param(p, table, builder) for p in item.fields))
    if isinstance(item, syntax.EnumDef):
        return replace(
            item,
            variants=tuple(
                replace(
                    v,
                    fields=tuple(_rewrite_param(p, table, builder) for p in v.fields),
                )
                for v in item.variants
            ),
        )
    if isinstance(item, syntax.ExceptionDef):
        return replace(item, fields=tuple(_rewrite_param(p, table, builder) for p in item.fields))
    if isinstance(item, syntax.ParamDecl):
        return replace(
            item,
            default=None if item.default is None else _rewrite_expr(item.default, table, builder),
        )
    return item


def _rewrite_param(
    param: syntax.Param,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.Param:
    if param.default is None:
        return param
    return replace(param, default=_rewrite_expr(param.default, table, builder))


def _rewrite_assign_target(
    target: syntax.AssignTarget,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.AssignTarget:
    if isinstance(target, syntax.IndexTarget):
        return replace(
            target,
            obj=_rewrite_expr(target.obj, table, builder),
            index=_rewrite_expr(target.index, table, builder),
        )
    return target


def _rewrite_expr(
    expr: syntax.Expr | _RawInfixChain,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.Expr:
    if isinstance(expr, _RawInfixChain):
        return _resolve_infix_chain(expr, table, builder)
    if isinstance(expr, syntax.FieldAccess):
        return replace(expr, obj=_rewrite_expr(expr.obj, table, builder))
    if isinstance(expr, syntax.IndexAccess):
        return replace(
            expr,
            obj=_rewrite_expr(expr.obj, table, builder),
            index=_rewrite_expr(expr.index, table, builder),
        )
    if isinstance(expr, syntax.Template):
        return replace(
            expr,
            segments=tuple(_rewrite_template_segment(s, table, builder) for s in expr.segments),
        )
    if isinstance(expr, syntax.UnaryNeg):
        return replace(expr, operand=_rewrite_expr(expr.operand, table, builder))
    if isinstance(expr, syntax.Cast):
        return replace(expr, expr=_rewrite_expr(expr.expr, table, builder))
    if isinstance(expr, syntax.IsTest):
        return replace(expr, expr=_rewrite_expr(expr.expr, table, builder))
    if isinstance(expr, syntax.TypeApply):
        return replace(expr, expr=_rewrite_expr(expr.expr, table, builder))
    if isinstance(expr, syntax.Call):
        return replace(
            expr,
            callee=_rewrite_expr(expr.callee, table, builder),
            args=tuple(_rewrite_expr(arg, table, builder) for arg in expr.args),
            named_args=tuple(_rewrite_named_arg(arg, table, builder) for arg in expr.named_args),
        )
    if isinstance(expr, syntax.Lambda):
        return replace(
            expr,
            params=tuple(_rewrite_param(p, table, builder) for p in expr.params),
            body=_rewrite_expr(expr.body, table, builder),
        )
    if isinstance(expr, syntax.Block):
        return _rewrite_block_infix(expr, table, builder)
    if isinstance(expr, syntax.If):
        return replace(
            expr,
            branches=tuple(_rewrite_if_branch(b, table, builder) for b in expr.branches),
        )
    if isinstance(expr, syntax.Case):
        return replace(
            expr,
            subject=_rewrite_expr(expr.subject, table, builder),
            branches=tuple(_rewrite_case_branch(b, table, builder) for b in expr.branches),
        )
    if isinstance(expr, syntax.Loop):
        return syntax.Loop(
            for_var=expr.for_var,
            for_iter=(
                None if expr.for_iter is None else _rewrite_expr(expr.for_iter, table, builder)
            ),
            for_range_to=(
                None
                if expr.for_range_to is None
                else _rewrite_expr(expr.for_range_to, table, builder)
            ),
            for_range_down=expr.for_range_down,
            for_range_by=(
                None
                if expr.for_range_by is None
                else _rewrite_expr(expr.for_range_by, table, builder)
            ),
            while_cond=(
                None if expr.while_cond is None else _rewrite_expr(expr.while_cond, table, builder)
            ),
            bound=None if expr.bound is None else _rewrite_expr(expr.bound, table, builder),
            body=_rewrite_expr(expr.body, table, builder),
            until_cond=(
                None if expr.until_cond is None else _rewrite_expr(expr.until_cond, table, builder)
            ),
            span=expr.span,
            node_id=expr.node_id,
        )
    if isinstance(expr, syntax.Try):
        return replace(
            expr,
            body=_rewrite_expr(expr.body, table, builder),
            handlers=tuple(_rewrite_catch_clause(h, table, builder) for h in expr.handlers),
        )
    if isinstance(expr, syntax.Raise):
        return replace(expr, exc=_rewrite_expr(expr.exc, table, builder))
    if isinstance(expr, syntax.Return):
        return replace(
            expr,
            value=None if expr.value is None else _rewrite_expr(expr.value, table, builder),
        )
    if isinstance(expr, syntax.ListLit):
        return replace(
            expr,
            elements=tuple(_rewrite_expr(element, table, builder) for element in expr.elements),
        )
    if isinstance(expr, syntax.DictLit):
        return replace(
            expr,
            entries=tuple(_rewrite_dict_entry(entry, table, builder) for entry in expr.entries),
        )
    return expr


def _rewrite_template_segment(
    segment: syntax.TemplateSegment,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.TemplateSegment:
    if isinstance(segment, syntax.InterpSegment):
        return replace(segment, expr=_rewrite_expr(segment.expr, table, builder))
    return segment


def _rewrite_named_arg(
    arg: syntax.NamedArg,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.NamedArg:
    return replace(arg, value=_rewrite_expr(arg.value, table, builder))


def _rewrite_if_branch(
    branch: syntax.IfBranch,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.IfBranch:
    cond: syntax.Expr | syntax.ElseSentinel
    if isinstance(branch.cond, syntax.ElseSentinel):
        cond = branch.cond
    else:
        cond = _rewrite_expr(branch.cond, table, builder)
    return replace(branch, cond=cond, body=_rewrite_expr(branch.body, table, builder))


def _rewrite_case_branch(
    branch: syntax.CaseBranch,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.CaseBranch:
    return replace(branch, body=_rewrite_expr(branch.body, table, builder))


def _rewrite_catch_clause(
    clause: syntax.CatchClause,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.CatchClause:
    return replace(clause, body=_rewrite_expr(clause.body, table, builder))


def _rewrite_dict_entry(
    entry: syntax.DictEntry,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.DictEntry:
    return replace(entry, value=_rewrite_expr(entry.value, table, builder))


def _resolve_infix_chain(
    chain: _RawInfixChain,
    table: dict[str, tuple[int, syntax.InfixAssoc, syntax.BinOp | None]],
    builder: AstBuilder,
) -> syntax.Expr:
    operands = [_rewrite_expr(operand.expr, table, builder) for operand in chain.operands]
    not_counts = [operand.not_count for operand in chain.operands]
    operand_spans = [operand.span for operand in chain.operands]
    operators = list(chain.operators)

    def parse_prefix(operand_index: int) -> tuple[syntax.Expr, int]:
        if not_counts[operand_index] > 0:
            not_counts[operand_index] -= 1
            operand, next_operand_index = parse_at(_NOT_PRIORITY, operand_index)
            return (
                syntax.UnaryNot(
                    operand=operand,
                    span=operand_spans[operand_index],
                    node_id=builder._next_id(),
                ),
                next_operand_index,
            )
        return operands[operand_index], operand_index + 1

    def parse_at(min_priority: int, operand_index: int) -> tuple[syntax.Expr, int]:
        left, next_operand_index = parse_prefix(operand_index)
        op_index = next_operand_index - 1
        while op_index < len(operators):
            op = operators[op_index]
            spec = table.get(op.name)
            if spec is None:
                raise AglSyntaxError(
                    f"Operator '{op.name}' must be declared with infixl or infixr before use.",
                    span=op.span,
                )
            priority, assoc, builtin = spec
            if priority < min_priority:
                break
            next_min = priority + 1 if assoc is syntax.InfixAssoc.LEFT else priority
            right, next_index = parse_at(next_min, op_index + 1)
            left = _make_infix_node(left, op, right, builtin, builder)
            next_operand_index = next_index
            op_index = next_operand_index - 1
        return left, next_operand_index

    result, final_index = parse_at(0, 0)
    assert final_index == len(operands)
    return result


def _make_infix_node(
    left: syntax.Expr,
    op: _InfixOperator,
    right: syntax.Expr,
    builtin: syntax.BinOp | None,
    builder: AstBuilder,
) -> syntax.Expr:
    span = _span_covering(left.span, right.span)
    if builtin is not None:
        if op.name in _NON_ASSOC_INFIX and (
            _is_nonassoc_binary(left) or _is_nonassoc_binary(right)
        ):
            raise AglSyntaxError(
                "Comparisons are non-associative; parenthesize explicitly, e.g. `(x == y) == z`.",
                span=op.span,
            )
        return syntax.BinaryOp(
            op=builtin,
            left=left,
            right=right,
            span=span,
            node_id=builder._next_id(),
        )
    callee = syntax.VarRef(name=op.name, span=op.span, node_id=builder._next_id())
    return syntax.Call(
        callee=callee,
        args=(left, right),
        named_args=(),
        span=span,
        node_id=builder._next_id(),
    )


def _is_nonassoc_binary(expr: syntax.Expr) -> bool:
    return isinstance(expr, syntax.BinaryOp) and expr.op.value in _NON_ASSOC_INFIX


def _span_covering(left: SourceSpan, right: SourceSpan) -> SourceSpan:
    return SourceSpan(
        start_line=left.start_line,
        start_col=left.start_col,
        end_line=right.end_line,
        end_col=right.end_col,
        start_offset=left.start_offset,
        end_offset=right.end_offset,
        source=left.source,
    )


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


def _type_expr_spelling(t: TypeExpr) -> str:
    """Return the source-level spelling of a primitive TypeExpr.

    Used to produce user-facing error messages that cite the source token text
    (e.g. ``'int'``) rather than an internal class name (e.g. ``'IntT'``).
    Only primitive / simple types are handled; complex types fall back to the
    class name (without the trailing ``T``).
    """
    _SPELLING: dict[type, str] = {
        TextT: "text",
        JsonT: "json",
        BoolT: "bool",
        IntT: "int",
        DecimalT: "decimal",
        UnitT: "unit",
        AgentT: "agent",
    }
    spelling = _SPELLING.get(type(t))
    if spelling is not None:
        return spelling
    if isinstance(t, NameT):
        return t.name
    # Fallback for complex types (ListT, DictT, FuncT, AppliedT): strip trailing 'T'.
    cls = type(t).__name__
    return cls[:-1].lower() if cls.endswith("T") else cls.lower()


def syntax_error_from_meta(meta: Meta, message: str) -> AglSyntaxError:
    """Create an AglSyntaxError from a Meta object."""
    return AglSyntaxError(message, span=_span_from_meta(meta))
