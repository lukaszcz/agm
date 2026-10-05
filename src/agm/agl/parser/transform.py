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
from typing import Iterable, TypeAlias, TypeGuard, TypeVar, cast

from lark import Transformer, v_args
from lark.lexer import Token
from lark.tree import Meta

import agm.agl.syntax as syntax
from agm.agl.constraints import CONSTRAINT_SPELLINGS
from agm.agl.infix import BUILTIN_FIXITIES, BUILTIN_OPS, group_infix, groups_at_parse
from agm.agl.lexer.positions import token_span
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.syntax.nodes import ELSE
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceId, SourceSpan, span_covering
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
    TypeExpr,
    UnitT,
)
from agm.util.decimal import decimal_in_range

# The receiver parameter every method declares first.
_SELF_PARAM = "self"

# An environment hole ``${NAME}`` reads the environment through this function,
# named in full so that neither an import nor a local binding can affect it.
_ENVIRONMENT_MODULE = "std/prelude"
_ENVIRONMENT_READER = "getenv"


@dataclass(frozen=True, slots=True)
class _RawPlaceholder:
    """Transformer-internal placeholder argument before call-level validation."""

    raw_digits: str | None
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _ScopePath:
    """Transformer-internal path retaining each segment's span."""

    segments: tuple[tuple[str, SourceSpan], ...]

    @property
    def spelling(self) -> str:
        return "::".join(name for name, _span in self.segments)


@dataclass(frozen=True, slots=True)
class _SelectedAtom:
    """A selected member before it is materialized as an AST item."""

    path: _ScopePath
    rename: str | None
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class _Braces:
    """A brace group retained until its enclosing declaration validates it."""

    entries: tuple[_SelectedAtom | _Braces | Token, ...]


@dataclass(frozen=True, slots=True)
class _Selection:
    """A declaration tail before source-form validation."""

    braces: _Braces | None = None
    atom: _SelectedAtom | None = None

    @property
    def glob(self) -> bool:
        """A selection that names neither braces nor an atom is a glob."""
        return self.braces is None and self.atom is None


# Import and export selection items share a field set, so one builder serves
# both; the value restriction keeps each instantiation concrete.
_SelectionItemT = TypeVar("_SelectionItemT", syntax.ImportItem, syntax.ExportItem)


_SCOPED_DECLARATIONS = (
    syntax.FuncDef,
    syntax.RecordDef,
    syntax.EnumDef,
    syntax.ExceptionDef,
    syntax.TypeAlias,
    syntax.LetDecl,
    syntax.VarDecl,
    syntax.ImportDecl,
    syntax.ExportDecl,
    syntax.UseDecl,
    syntax.BuiltinVarDecl,
)

# Message for the decl_head module-route rejection. Also used for a `var`
# binder path, which reuses `decl_head`; a `let` binder path never raises
# this — a chain that fails the spellable-as-a-declaration-head test simply
# keeps its constructor-pattern meaning instead.
_MODULE_ROUTE_MESSAGE = "scope paths use '::' between name segments."

#: What separates the segments of a module route and, written first, anchors it.
_ROUTE_SEPARATOR = "/"

#: What joins a qualifier to what follows it. The lexer drops it from a merged
#: qualifier's value but not from its span.
_QUALIFIER_DELIMITER = "::"


@dataclass(frozen=True, slots=True)
class _RouteToken:
    """One lexer-merged route token, split into the parts a builder needs."""

    name: str
    anchored: bool
    span: SourceSpan


def _prefix_scope_path(
    item: syntax.ScopeItem, prefix: tuple[syntax.ScopeSegment, ...]
) -> syntax.ScopeItem:
    """Add *prefix* to every declaration path contained by a scope item."""
    if isinstance(item, syntax.ScopeRegion):
        return replace(item, items=tuple(_prefix_scope_path(child, prefix) for child in item.items))
    return replace(item, scope_path=prefix + item.scope_path)


# ---------------------------------------------------------------------------
# Transformer-internal markers and provenance (never leaks into the AST)
# ---------------------------------------------------------------------------


_InfixOperator: TypeAlias = syntax.RawInfixOperator
_InfixOperand: TypeAlias = syntax.RawInfixOperand
_RawInfixChain: TypeAlias = syntax.RawInfixChain


@dataclass(frozen=True, slots=True)
class _InfixPriority:
    """Transformer-internal priority specification for ``infix`` declarations."""

    value: int | None
    base: str | None
    delta: int


@dataclass(frozen=True, slots=True)
class _RawNamedArg:
    """Transformer-internal named argument whose value may be a placeholder."""

    name: str
    value: _RawPlaceholder
    span: SourceSpan


_RawPosArg: TypeAlias = syntax.Expr | _RawPlaceholder | _RawInfixChain
_RawNamed: TypeAlias = syntax.NamedArg[syntax.Expr] | _RawNamedArg
_RawArgLists: TypeAlias = tuple[list[_RawPosArg], list[_RawNamed]]
_CallArgs: TypeAlias = tuple[
    tuple[syntax.CallArg, ...], tuple[syntax.NamedArg[syntax.CallArg], ...]
]
_ArgLists: TypeAlias = tuple[list[syntax.CallArg], list[syntax.NamedArg[syntax.CallArg]]]
_JuxtCall: TypeAlias = tuple[tuple[TypeExpr, ...], _ArgLists]
_RawAttrPosArg: TypeAlias = syntax.Expr | _RawInfixChain
_RawAttrArgLists: TypeAlias = tuple[list[_RawAttrPosArg], list[syntax.AttributeKeyedArg]]
_RawItem: TypeAlias = syntax.Item | _RawInfixChain
#: One ``use`` target before its spelling is resolved: the merged qualifier
#: prefixes, the final target name, a whole-target alias, and the tail.
_UseTarget: TypeAlias = tuple[tuple[Token, ...], Token | None, str | None, _Selection | None]


@dataclass(frozen=True, slots=True)
class _JuxtSuffix:
    """A deferred juxtaposition postfix operation allocated in source order."""

    kind: str
    value: str | syntax.Expr | _JuxtCall
    span: SourceSpan
    node_id: int


@dataclass(frozen=True, slots=True)
class _JuxtField:
    """A juxtaposition field token and its already allocated node id."""

    name: str
    span: SourceSpan
    node_id: int


@dataclass(frozen=True, slots=True)
class _QualifierChainSegment:
    """Transformer-internal qualifier segment retaining its leading anchor."""

    segment: syntax.QualifierSegment
    anchored: bool = False


@dataclass(frozen=True, slots=True)
class _BuiltinReceiverHead:
    """A function declaration head headed by an applied builtin receiver."""

    name: str
    receiver_scope_name: str
    receiver_type: TypeExpr
    span: SourceSpan


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
    ArrayT,
    DictT,
    UnitT,
    FuncT,
    AppliedT,
)

# The concrete pattern AST node types (used to pick a sub-pattern out of rule children).
_PATTERN_NODE_TYPES = (
    syntax.WildcardPattern,
    syntax.LiteralPattern,
    syntax.VarPattern,
    syntax.AsPattern,
    syntax.ConstructorPattern,
)


def _is_str_tuple(a: object) -> bool:
    """Return True iff *a* is a non-empty tuple whose elements are all ``str``."""
    return isinstance(a, tuple) and len(a) > 0 and all(isinstance(x, str) for x in a)


def _is_stray_scope_end(item: syntax.Item) -> bool:
    """Whether a root item has the unpromoted spelling of a stray closer."""
    return (
        isinstance(item, syntax.Call)
        and isinstance(item.callee, syntax.VarRef)
        and item.callee.name == "end"
        and len(item.args) == 1
        and isinstance(item.args[0], syntax.VarRef)
        and not item.named_args
    )


# Names for region items that remain disallowed, keyed by node type. A
# single, obvious table so admitting a form later is a table edit rather
# than hunting down a scattered isinstance branch.
_REJECTED_SCOPE_ITEM_NAMES: tuple[tuple[type, str], ...] = (
    (syntax.AssignStmt, "an assignment"),
    (syntax.InfixDecl, "an 'infix' declaration"),
)


# ``scope_item ::= scope_region | declaration | binder | expr``; once the
# admitted shapes are filtered out, only these remain.
_RejectedScopeItem: TypeAlias = _RawInfixChain | syntax.InfixDecl | syntax.AssignStmt | syntax.Expr


def _rejected_scope_item(item: _RejectedScopeItem) -> tuple[str, SourceSpan]:
    """Name and locate a disallowed scope-region item, for the diagnostic.

    ``scope_item ::= scope_region | declaration | binder | expr``, and the
    caller has already filtered out every admitted shape (``allowed_items``),
    so a disallowed item is always one of the forms in this signature — each
    of which carries its own span.
    """
    for node_type, form in _REJECTED_SCOPE_ITEM_NAMES:
        if isinstance(item, node_type):
            return form, item.span
    return "a bare expression", item.span


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
        allow_late_uses: bool = False,
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
        # Transcript parsing discovers entry boundaries only; each entry is
        # parsed normally before evaluation and owns its own header ordering.
        self._allow_late_uses = allow_late_uses

    def _source_start_span(self) -> SourceSpan:
        """Build the empty span at the start of the source being built.

        An empty module root matches no tokens, so Lark leaves its meta unset;
        this is the position such a module and its items-free block report.
        """
        return SourceSpan(
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=1,
            start_offset=0,
            end_offset=0,
            source=self._source,
        )

    def _span_from_meta(self, meta: Meta) -> SourceSpan:
        """Build a SourceSpan from Lark tree Meta, stamped with self._source.

        A rule that matched no tokens carries no position, and spans the start
        of its source instead.
        """
        if meta.empty:
            return self._source_start_span()
        return SourceSpan(
            start_line=meta.line,
            start_col=meta.column,
            end_line=meta.end_line,
            end_col=meta.end_column,
            start_offset=meta.start_pos,
            end_offset=meta.end_pos,
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
        block = cast(syntax.Block, args[0])
        stray_end = next((item for item in block.items if _is_stray_scope_end(item)), None)
        if stray_end is not None:
            raise AglSyntaxError("stray 'end'; no scope region is open.", span=stray_end.span)
        span = self._span_from_meta(meta)
        return syntax.Program(body=block, span=span, node_id=self._next_id())

    def _build_block(self, meta: Meta, args: _Args) -> syntax.Block:
        """Build a root or suite block from its non-layout children."""
        items = tuple(
            cast(_RawItem, arg) for arg in args if arg is not None and not isinstance(arg, Token)
        )
        return syntax.Block(
            items=cast(tuple[syntax.Item, ...], items),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def module_block(self, meta: Meta, args: _Args) -> syntax.Block:
        """Build the module-root block, whose items may include scope regions.

        It is also the only block that may be empty: a source holding no items
        is a legal, empty module.
        """
        block = self._build_block(meta, args)
        if not self._allow_late_uses:
            self._validate_use_placement(block.items)
        return block

    def _validate_use_placement(self, items: tuple[syntax.Item, ...]) -> None:
        """Require use declarations to remain in the header portion of a block.

        Import and export placement is not this check's concern -- the scope
        pass owns that rule uniformly for every module root and region. This
        check only enforces that use declarations precede other items, while
        treating import and export items as header items. Each block is
        validated independently as its transform is built bottom-up.
        """
        seen_non_header = False
        for item in items:
            if isinstance(item, syntax.UseDecl):
                if seen_non_header:
                    raise AglSyntaxError(
                        "use declarations must precede non-header items.",
                        span=item.span,
                    )
            elif not isinstance(item, (syntax.ImportDecl, syntax.ExportDecl)):
                seen_non_header = True

    def block(self, meta: Meta, args: _Args) -> syntax.Block:
        """Build a regular suite block, which cannot contain scope regions or uses."""
        block = self._build_block(meta, args)
        misplaced_use = next(
            (item for item in block.items if isinstance(item, syntax.UseDecl)), None
        )
        if misplaced_use is not None:
            raise AglSyntaxError(
                "use declarations are only allowed at module root or in scope regions.",
                span=misplaced_use.span,
            )
        return block

    # ------------------------------------------------------------------
    # Scope regions
    # ------------------------------------------------------------------

    def _path_from_tokens(self, args: _Args) -> _ScopePath:
        """Build a scope path from names and lexer-merged qualifier segments.

        A qualifier segment spelling a module route belongs to no scope path,
        so it is rejected here rather than resolved as a name containing a
        separator.
        """
        segments: list[tuple[str, SourceSpan]] = []
        for arg in cast(list[Token], args):
            if _is_typed_token(arg, "MODQUAL"):
                segments.append(self._scope_segment(arg))
            else:
                segments.append((str(arg), token_span(arg, self._source)))
        return _ScopePath(segments=tuple(segments))

    def scope_path(self, meta: Meta, args: _Args) -> _ScopePath:
        """Build a scope path, rejecting slash-separated module routes."""
        return self._path_from_tokens(args)

    def decl_head(self, meta: Meta, args: _Args) -> _ScopePath:
        """Build a declaration head with an optional scope-path prefix."""
        return self._path_from_tokens(args)

    def path_atom(self, meta: Meta, args: _Args) -> _ScopePath:
        """Build a module-selection path atom."""
        return self._path_from_tokens(args)

    def _scope_segments(self, path: _ScopePath) -> tuple[syntax.ScopeSegment, ...]:
        return tuple(
            syntax.ScopeSegment(name=name, span=span, node_id=self._next_id())
            for name, span in path.segments
        )

    def _declaration_head(self, args: _Args) -> tuple[str, tuple[syntax.ScopeSegment, ...]]:
        path = next(arg for arg in args if isinstance(arg, _ScopePath))
        name, _span = path.segments[-1]
        return name, self._scope_segments(_ScopePath(path.segments[:-1]))

    def _receiver_type_params(self, receiver: TypeExpr | None) -> tuple[str, ...]:
        """Return the positional type-variable slots bound by an applied receiver.

        Any applied head (``Box[T]``) contributes its bare-name arguments;
        whether it is a real builtin receiver is scope's concern.
        """
        if isinstance(receiver, ArrayT):
            children: tuple[TypeExpr, ...] = (receiver.elem,)
        elif isinstance(receiver, DictT):
            children = (receiver.key, receiver.value)
        elif isinstance(receiver, AppliedT):
            children = receiver.args
        else:
            children = ()
        return tuple(child.name for child in children if isinstance(child, NameT))

    def _function_declaration_head(
        self, args: _Args
    ) -> tuple[str, tuple[syntax.ScopeSegment, ...], TypeExpr | None]:
        receiver = next((arg for arg in args if isinstance(arg, _BuiltinReceiverHead)), None)
        if receiver is None:
            name, scope_path = self._declaration_head(args)
            return name, scope_path, None
        return (
            receiver.name,
            (
                syntax.ScopeSegment(
                    name=receiver.receiver_scope_name,
                    span=receiver.span,
                    node_id=self._next_id(),
                ),
            ),
            receiver.receiver_type,
        )

    def scope_region(self, meta: Meta, args: _Args) -> syntax.ScopeRegion:
        """Build and normalize a scope region with a matching closer."""
        paths = [arg for arg in args if isinstance(arg, _ScopePath)]
        header, closer = paths
        if tuple(name for name, _span in closer.segments) != tuple(
            name for name, _span in header.segments
        ):
            closer_span = closer.segments[0][1]
            raise AglSyntaxError(
                f"scope closer does not match; expected 'end {header.spelling}'.",
                span=closer_span,
            )

        allowed_items = (syntax.ScopeRegion, syntax.UseDecl, *_SCOPED_DECLARATIONS)
        items = tuple(arg for arg in args if isinstance(arg, allowed_items))
        self._validate_use_placement(cast(tuple[syntax.Item, ...], items))
        disallowed = next(
            (
                arg
                for arg in args
                if arg is not None and not isinstance(arg, (Token, _ScopePath, allowed_items))
            ),
            None,
        )
        if disallowed is not None:
            form, span = _rejected_scope_item(cast(_RejectedScopeItem, disallowed))
            raise AglSyntaxError(
                f"scope regions cannot contain {form}.",
                span=span,
            )

        segments = self._scope_segments(header)
        *outer, innermost = segments
        span = self._span_from_meta(meta)
        region = syntax.ScopeRegion(
            segment=innermost, items=items, span=span, node_id=self._next_id()
        )
        for segment in reversed(outer):
            region = syntax.ScopeRegion(
                segment=segment, items=(region,), span=span, node_id=self._next_id()
            )
        return cast(syntax.ScopeRegion, _prefix_scope_path(region, segments))

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def builtin_var_def(self, meta: Meta, args: _Args) -> syntax.BuiltinVarDecl:
        """builtin_var_def: attributes? "builtin" _NEWLINE? VAR name type_ann (EQ expr)?"""
        rest = _without_attributes(args)
        name_tok = _find_name_token(rest)
        type_expr = _find_type_expr(rest[1:])
        default = cast(
            syntax.Expr,
            next((arg for arg in rest if _is_expr_node(arg)), None),
        )
        return syntax.BuiltinVarDecl(
            name=str(name_tok),
            type_ann=type_expr,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            default=default,
            attributes=_find_attributes(args),
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
        tok = _required_token(args, "INT")
        return _InfixPriority(value=int(str(tok)), base=None, delta=0)

    def infix_priority_relative(self, meta: Meta, args: _Args) -> _InfixPriority:
        self._require_infix_at_keyword(meta, args)
        op = next(a for a in args if isinstance(a, _InfixOperator))
        delta = next((a for a in args if isinstance(a, int)), 0)
        return _InfixPriority(value=None, base=op.name, delta=delta)

    def _require_infix_at_keyword(self, meta: Meta, args: _Args) -> None:
        first_name = _typed_token(args, "NAME")
        if first_name is None or str(first_name) != "at":
            raise syntax_error_from_meta(meta, "infix priority must start with 'at'.")

    def priority_delta_plus(self, meta: Meta, args: _Args) -> int:
        tok = _required_token(args, "INT")
        return int(str(tok))

    def priority_delta_minus(self, meta: Meta, args: _Args) -> int:
        tok = _required_token(args, "INT")
        return -int(str(tok))

    # ------------------------------------------------------------------
    # record_def / field_def
    # ------------------------------------------------------------------

    def record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        # Grammar: "record" name type_params? (EQ? record_body)?
        name, scope_path = self._declaration_head(args)
        type_params_val = _find_type_params(args)
        attributes = _find_attributes(args)
        return syntax.RecordDef(
            name=name,
            fields=_find_field_tuple(args),
            type_param_slots=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=attributes,
        )

    # ------------------------------------------------------------------
    # Declaration attributes
    # ------------------------------------------------------------------

    def attributes(self, meta: Meta, args: _Args) -> tuple[syntax.Attribute, ...]:
        """attributes: attribute+ — the prefix written in front of one target."""
        return tuple(a for a in args if isinstance(a, syntax.Attribute))

    def attribute(self, meta: Meta, args: _Args) -> syntax.Attribute:
        """attribute: AT NAME (LPAR attr_arg_list? RPAR)? — kept exactly as written."""
        name_tok = next(a for a in args if _is_name_token(a))
        span = self._span_from_meta(meta)
        raw_pos: list[_RawAttrPosArg] = []
        keyed_args: list[syntax.AttributeKeyedArg] = []
        for a in args:
            if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], list):
                raw_pos, keyed_args = cast(_RawAttrArgLists, a)
        return syntax.Attribute(
            name=str(name_tok),
            args=tuple(cast(syntax.Expr, a) for a in raw_pos),
            keyed_args=tuple(keyed_args),
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # record_def / field_def (continued)
    # ------------------------------------------------------------------

    def record_indent_body(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: _field_block
        return tuple(a for a in args if isinstance(a, syntax.Param))

    def record_paren_body(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: LPAR field_list? RPAR
        return next((cast(tuple[syntax.Param, ...], a) for a in args if _is_field_tuple(a)), ())

    record_inline_body = record_paren_body

    def field_def(self, meta: Meta, args: _Args) -> syntax.Param:
        # Grammar: attributes? VAR? field_name COLON type_expr (EQ or_expr)?
        rest = _without_attributes(args)
        name_tok = _find_name_token(rest)
        type_expr, default = _extract_ann_and_optional_expr(rest)
        return syntax.Param(
            name=str(name_tok),
            type_expr=type_expr,
            default=default,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            mutable=any(isinstance(arg, Token) and arg.type == "VAR" for arg in rest),
            attributes=_find_attributes(args),
        )

    # ------------------------------------------------------------------
    # enum_def / variant_def / variant_payload / field_list / field_inline
    # ------------------------------------------------------------------

    def enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        # Grammar: "enum" name type_params? EQ? enum_body
        name, scope_path = self._declaration_head(args)
        type_params_val = _find_type_params(args)
        members = _find_member_tuple(args)
        return syntax.EnumDef(
            name=name,
            members=members,
            type_param_slots=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=_find_attributes(args),
        )

    def enum_body(
        self, meta: Meta, args: _Args
    ) -> tuple[syntax.VariantDef | syntax.VariantRef, ...]:
        return _find_member_tuple(args)

    def enum_variant_seq(
        self, meta: Meta, args: _Args
    ) -> tuple[syntax.VariantDef | syntax.VariantRef, ...]:
        return tuple(a for a in args if isinstance(a, (syntax.VariantDef, syntax.VariantRef)))

    def variant_def(self, meta: Meta, args: _Args) -> syntax.VariantDef:
        # Grammar: PIPE? attributes? name variant_payload?
        name_tok = next(a for a in args if _is_name_token(a))
        fields: tuple[syntax.Param, ...] = ()
        for a in args:
            if _is_field_tuple(a):
                fields = cast(tuple[syntax.Param, ...], a)
                break
        attributes = _find_attributes(args)
        return syntax.VariantDef(
            name=str(name_tok),
            fields=fields,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            attributes=attributes,
        )

    def member_type_args(self, meta: Meta, args: _Args) -> tuple[TypeExpr, ...]:
        """Return a referenced member's ordinary bracketed type arguments."""
        return next(
            (
                cast(tuple[TypeExpr, ...], arg)
                for arg in args
                if isinstance(arg, tuple) and (not arg or isinstance(arg[0], _ALL_TYPE_EXPRS))
            ),
            (),
        )

    def variant_ref(self, meta: Meta, args: _Args) -> syntax.VariantRef:
        chain = next(arg for arg in args if isinstance(arg, syntax.QualifierChain))
        type_args = next(
            (
                arg
                for arg in args
                if isinstance(arg, tuple) and (not arg or isinstance(arg[0], _ALL_TYPE_EXPRS))
            ),
            (),
        )
        return syntax.VariantRef(
            chain=chain,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            type_args=cast(tuple[TypeExpr, ...], type_args),
        )

    def variant_ref_with_payload(self, meta: Meta, args: _Args) -> syntax.VariantRef:
        raise AglSyntaxError(
            "A member reference may not carry a field list; the shape comes from the record.",
            span=self._span_from_meta(meta),
        )

    variant_paren_payload = record_paren_body
    variant_indent_payload = record_indent_body

    def field_list(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        # Grammar: field_inline (COMMA field_inline)* COMMA?
        return tuple(a for a in args if isinstance(a, syntax.Param))

    # Grammar: attributes? VAR? field_name COLON type_expr (EQ or_expr)? — same shape as
    # ``field_def``.
    field_inline = field_def

    # ------------------------------------------------------------------
    # exception_def / exception_body
    # ------------------------------------------------------------------

    def exception_def(self, meta: Meta, args: _Args) -> syntax.ExceptionDef:
        # Grammar: "exception" name exception_base? exception_body?
        name, scope_path = self._declaration_head(args)
        base = next((a for a in args if type(a) is str), None)
        attributes = _find_attributes(args)
        return syntax.ExceptionDef(
            name=name,
            fields=_find_field_tuple(args),
            base=base,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=attributes,
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
        name, scope_path = self._declaration_head(args)
        type_params_val = _find_type_params(args)
        type_expr = _find_type_expr(args)
        return syntax.TypeAlias(
            name=name,
            type_expr=type_expr,
            type_param_slots=type_params_val,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=_find_attributes(args),
        )

    # ------------------------------------------------------------------
    # func_def / param_list / param_def / func_body
    # ------------------------------------------------------------------

    def _func_def(
        self,
        meta: Meta,
        args: _Args,
        *,
        is_program: bool = False,
        is_builtin: bool = False,
        is_extern: bool = False,
    ) -> syntax.FuncDef:
        """Build a function definition from any ``def`` form.

        ``def`` and ``program def`` carry a body; ``builtin def`` and ``extern
        def`` share the same shape but declare a return type in place of one.
        """
        name, scope_path, receiver_type = self._function_declaration_head(args)
        params, return_type = self._split_params_and_type(args)
        _check_function_param_annotations(params)
        # ``builtin``/``extern`` forms end at their return type; the others at a body.
        last = args[-1]
        body = None if isinstance(last, _ALL_TYPE_EXPRS) else cast(syntax.Expr, last)
        return syntax.FuncDef(
            name=name,
            params=params,
            return_type=return_type,
            body=body,
            type_param_slots=self._receiver_type_params(receiver_type) + _find_type_params(args),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            is_program=is_program,
            is_builtin=is_builtin,
            is_extern=is_extern,
            is_method=_declares_method(params),
            scope_path=scope_path,
            receiver_type=receiver_type,
            attributes=_find_attributes(args),
            constraints=_find_constraints(args),
        )

    def func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """func_def: "def" func_def_tail"""
        return self._func_def(meta, args)

    def program_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """program_func_def: "program" "def" func_def_tail"""
        return self._func_def(meta, args, is_program=True)

    def func_decl_head(self, meta: Meta, args: _Args) -> object:
        """Unwrap an ordinary function declaration head."""
        return args[0]

    def builtin_receiver_head(self, meta: Meta, args: _Args) -> _BuiltinReceiverHead:
        """Build an applied builtin receiver declaration head."""
        segment = next(arg for arg in args if isinstance(arg, _QualifierChainSegment)).segment
        name = str(next(arg for arg in args if _is_name_token(arg)))
        type_args = cast(tuple[TypeExpr, ...], segment.type_args)
        if segment.name == "array" and len(type_args) == 1:
            receiver_type: TypeExpr = ArrayT(
                elem=type_args[0], span=segment.span, node_id=self._next_id()
            )
        elif segment.name == "dict" and len(type_args) == 2:
            receiver_type = DictT(
                key=type_args[0], value=type_args[1], span=segment.span, node_id=self._next_id()
            )
        else:
            receiver_type = AppliedT(
                name=segment.name,
                args=type_args,
                span=segment.span,
                node_id=self._next_id(),
            )
        return _BuiltinReceiverHead(
            name=name,
            receiver_scope_name=segment.name,
            receiver_type=receiver_type,
            span=segment.span,
        )

    def builtin_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """builtin_func_def: "builtin" "def" name type_params? (...) -> type_expr"""
        return self._func_def(meta, args, is_builtin=True)

    def extern_func_def(self, meta: Meta, args: _Args) -> syntax.FuncDef:
        """extern_func_def: "extern" "def" name type_params? (...) -> type_expr"""
        return self._func_def(meta, args, is_extern=True)

    def param_list(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        """param_list: param_def (COMMA param_def)* COMMA?"""
        return tuple(a for a in args if isinstance(a, syntax.Param))

    def param_def(self, meta: Meta, args: _Args) -> syntax.Param:
        """param_def: attributes? field_name (COLON type_expr)? (EQ or_expr)?"""
        rest = _without_attributes(args)
        name_tok = _find_name_token(rest)
        name = str(name_tok)
        type_expr, default = _extract_ann_and_optional_expr(rest[1:])
        return syntax.Param(
            name=name,
            type_expr=type_expr,
            default=default,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            attributes=_find_attributes(args),
        )

    def func_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """func_body: suite_expr | expr — pass through the inner expression."""
        # The argument is a Block from suite_expr or an expression from expr.
        expr = _find_non_token(args)
        return cast(syntax.Expr, expr)

    def _block_from_items(self, meta: Meta, args: _Args) -> syntax.Block:
        """Build a Block from every non-token argument, in order.

        Shared by the marked inline body sequences (loop, try, and
        parenthesized blocks), which differ only in their terminators.
        """
        items = tuple(
            cast(syntax.Item, a) for a in args if a is not None and not isinstance(a, Token)
        )
        return syntax.Block(
            items=items,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _body_as_expr(self, meta: Meta, item: _RawItem) -> syntax.Expr:
        """Coerce a single inline body item to an expression.

        A binder is not an expression, so it is wrapped in a single-item Block
        — the same shape the equivalent suite body produces.
        """
        if isinstance(item, (syntax.LetDecl, syntax.VarDecl, syntax.AssignStmt)):
            return syntax.Block(
                items=(item,),
                span=self._span_from_meta(meta),
                node_id=self._next_id(),
            )
        return cast(syntax.Expr, item)

    # ------------------------------------------------------------------
    # let_decl / var_decl / assign_stmt
    # ------------------------------------------------------------------

    def let_decl(self, meta: Meta, args: _Args) -> syntax.LetDecl:
        """let_decl: attributes? "let" decl_head type_ann? EQ expr"""
        rest = _without_attributes(args)
        name, scope_path = self._declaration_head(rest)
        ann, value = _extract_ann_and_value(rest[1:])
        span = self._span_from_meta(meta)
        return syntax.LetDecl(
            name=name,
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=_find_attributes(args),
        )

    def var_decl(self, meta: Meta, args: _Args) -> syntax.VarDecl:
        """var_decl: attributes? VAR decl_head type_ann? EQ expr"""
        rest = _without_attributes(args)
        name, scope_path = self._declaration_head(rest)
        ann, value = _extract_ann_and_value(rest[1:])
        span = self._span_from_meta(meta)
        return syntax.VarDecl(
            name=name,
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
            scope_path=scope_path,
            attributes=_find_attributes(args),
        )

    def assign_stmt(self, meta: Meta, args: _Args) -> syntax.AssignStmt:
        # Grammar: postfix ASSIGN expr
        lhs, value = (cast(syntax.Expr, a) for a in args if _is_expr_node(a))
        target: syntax.AssignTarget
        if isinstance(lhs, syntax.VarRef):
            target = syntax.NameTarget(
                name=lhs.name,
                span=lhs.span,
                node_id=self._next_id(),
                qualifier=self._assignment_qualifier(lhs),
            )
        elif isinstance(lhs, syntax.IndexAccess):
            # Indexed assignment is legal on any array/dict-typed expression
            # root -- a `let` binding, a param, a field, a call result, or a
            # qualified read -- since it mutates a container rather than
            # rebinding a name.
            target = syntax.IndexTarget(
                obj=lhs.obj,
                index=lhs.index,
                span=lhs.span,
                node_id=self._next_id(),
            )
        elif isinstance(lhs, syntax.FieldAccess):
            target = syntax.FieldTarget(
                obj=lhs.obj,
                field=lhs.field,
                span=lhs.span,
                node_id=self._next_id(),
            )
        else:
            raise AglSyntaxError(
                "assignment target must be a variable, indexed expression, or field.",
                span=lhs.span,
            )
        span = self._span_from_meta(meta)
        return syntax.AssignStmt(
            target=target,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def _assignment_qualifier(self, ref: syntax.VarRef) -> syntax.QualifierChain | None:
        """Return a simple expression chain for a qualified assignment target.

        A chain of any length is admitted here -- a module route (always one
        segment) and a local scope path (any number of segments, for a
        nested-region ``var``) are indistinguishable until resolution, which
        owns the actual accept/reject decision. Only a type-argument-applied
        segment is rejected on sight: no assignment target can carry one.
        """
        chain = ref.qualifier
        if chain is None:
            return None
        if any(segment.type_args is not None for segment in chain.segments):
            raise AglSyntaxError(
                "a qualified assignment target cannot apply type arguments to a qualifier "
                "segment; type-qualified constructor forms are not assignment targets.",
                span=ref.span,
            )
        return chain

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
        name = str(args[0])
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
        type_args = _find_type_args(args)
        span = self._span_from_meta(meta)
        nid = self._next_id()
        if name == "array":
            if len(type_args) == 1:
                return ArrayT(elem=type_args[0], span=span, node_id=nid)
            raise syntax_error_from_meta(meta, "array[] takes exactly one type argument")
        if name == "dict":
            if len(type_args) == 2:
                return DictT(key=type_args[0], value=type_args[1], span=span, node_id=nid)
            raise syntax_error_from_meta(meta, "dict[] takes exactly two type arguments")
        return AppliedT(name=name, args=type_args, span=span, node_id=nid)

    def qual_applied_type(self, meta: Meta, args: _Args) -> AppliedT:
        """qual_ref_chain type_lsqb type_arg_list RSQB — qualified application."""
        qualifier = next(a for a in args if isinstance(a, syntax.QualifierChain))
        type_args = _find_type_args(args)
        return AppliedT(
            name=qualifier.member,
            args=type_args,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            qualifier=qualifier,
        )

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

    def constraint_block(self, meta: Meta, args: _Args) -> tuple[syntax.Constraint, ...]:
        """constraint_block: (LBRACE | CALL_LBRACE) constraint (COMMA constraint)* COMMA? RBRACE"""
        return tuple(a for a in args if isinstance(a, syntax.Constraint))

    def constraint(self, meta: Meta, args: _Args) -> syntax.Constraint:
        """constraint: NAME name — the first NAME must spell a known constraint kind."""
        kind_tok, param_tok = args
        kind_name = str(kind_tok)
        kind = CONSTRAINT_SPELLINGS.get(kind_name)
        if kind is None:
            known = ", ".join(sorted(CONSTRAINT_SPELLINGS))
            raise AglSyntaxError(
                f"{kind_name!r} is not a recognized constraint; expected one of {known}.",
                span=self._span_from_meta(meta),
            )
        return syntax.Constraint(
            kind=kind,
            param=str(param_tok),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def func_type(self, meta: Meta, args: _Args) -> FuncT:
        """LPAR type_list? RPAR THIN_ARROW type_expr — function type (A, B) -> C."""
        param_types = next((a for a in args if isinstance(a, tuple)), ())
        return FuncT(
            params=cast(tuple[TypeExpr, ...], param_types),
            result=cast(TypeExpr, args[-1]),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def unary_func_type(self, meta: Meta, args: _Args) -> FuncT:
        """type_atom THIN_ARROW type_expr — function type A -> B."""
        param_type, _arrow, result_type = args
        return FuncT(
            params=(cast(TypeExpr, param_type),),
            result=cast(TypeExpr, result_type),
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
        """LPAR expr RPAR — the inner expression."""
        return cast(syntax.Expr | _RawInfixChain, args[1])

    # ------------------------------------------------------------------
    # Lambda expression
    # ------------------------------------------------------------------

    def _split_params_and_type(
        self, args: _Args
    ) -> tuple[tuple[syntax.Param, ...], TypeExpr | None]:
        """Return a func/lambda arg list's parameters and optional return type."""
        params = next((a for a in args if _is_field_tuple(a)), ())
        return_type = next((a for a in args if isinstance(a, _ALL_TYPE_EXPRS)), None)
        return cast(tuple[syntax.Param, ...], params), return_type

    def parenthesized_lambda_params(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        """Return the optional parameter list inside a lambda's parentheses."""
        return cast(tuple[syntax.Param, ...], next((a for a in args if _is_field_tuple(a)), ()))

    def unary_lambda_param(self, meta: Meta, args: _Args) -> tuple[syntax.Param, ...]:
        """Build the unannotated parameter of a bare unary lambda."""
        name_token = _find_name_token(args)
        return (
            syntax.Param(
                name=str(name_token),
                type_expr=None,
                default=None,
                span=self._span_from_meta(meta),
                node_id=self._next_id(),
            ),
        )

    def lambda_expr(self, meta: Meta, args: _Args) -> syntax.Lambda:
        """Build an anonymous function with parenthesized or bare unary parameters."""
        params, return_type = self._split_params_and_type(args)
        return syntax.Lambda(
            params=params,
            return_type=return_type,
            body=cast(syntax.Expr, args[-1]),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def leading_dot_expr(self, meta: Meta, args: _Args) -> syntax.Lambda:
        """Desugar ``.method(args)`` to a unary lambda over contextual ``self``."""
        span = self._span_from_meta(meta)
        name_token = _find_name_token(args)
        self_span = token_span(name_token, self._source)
        param = syntax.Param(
            name="self",
            type_expr=None,
            default=None,
            span=self_span,
            node_id=self._next_id(),
        )
        receiver = syntax.VarRef(name="self", span=self_span, node_id=self._next_id())
        member = syntax.FieldAccess(
            obj=receiver,
            field=str(name_token),
            span=span,
            node_id=self._next_id(),
        )
        pos_args, named_args = self._call_args_from_children(args, span)
        body = syntax.Call(
            callee=member,
            args=pos_args,
            named_args=named_args,
            span=span,
            node_id=self._next_id(),
            type_args=_find_type_args(args),
        )
        return syntax.Lambda(
            params=(param,),
            return_type=None,
            body=body,
            span=span,
            node_id=self._next_id(),
            implicit_self=True,
        )

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def lit_int(self, meta: Meta, args: _Args) -> syntax.IntLit:
        tok = cast(Token, args[0])
        return syntax.IntLit(
            value=int(str(tok)),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _build_decimal_lit(self, tok: Token, meta: Meta) -> syntax.DecimalLit:
        """Build a ``DecimalLit`` from *tok*, rejecting one outside the pinned range.

        Shared by :meth:`lit_decimal` and :meth:`pat_lit_decimal` -- the one
        place a decimal literal's text becomes a ``Decimal`` -- so both an
        expression literal and a pattern literal out of range are caught here,
        at compile time, with the same range predicate every other decimal
        creation site uses.
        """
        value = decimal.Decimal(str(tok))
        span = self._span_from_meta(meta)
        if not decimal_in_range(value):
            # The bound, not the value: an out-of-range literal can carry
            # well over a million digits, and *span* already identifies the
            # rejected source text.
            raise AglSyntaxError("decimal literal is out of range", span=span)
        return syntax.DecimalLit(value=value, span=span, node_id=self._next_id())

    def lit_decimal(self, meta: Meta, args: _Args) -> syntax.DecimalLit:
        tok = cast(Token, args[0])
        return self._build_decimal_lit(tok, meta)

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
        tok = cast(Token, args[0])
        return syntax.VarRef(
            name=str(tok), span=self._span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # Postfix: call / field_access / index_access
    # ------------------------------------------------------------------

    def _call_args_from_children(self, args: _Args, span: SourceSpan) -> _CallArgs:
        """Finalize the optional argument list among one call-like rule's children."""
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], list):
                raw_pos, raw_named = cast(_RawArgLists, arg)
                return self._finalize_call_args(raw_pos, raw_named, call_span=span)
        return (), ()

    def call(self, meta: Meta, args: _Args) -> syntax.Call:
        """postfix LPAR arg_list? RPAR → Call node."""
        callee, type_args = _split_type_apply(cast(syntax.Expr, args[0]))
        span = self._span_from_meta(meta)
        pos_args, named_args = self._call_args_from_children(args[1:], span)
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
        """postfix INDEX_LSQB expr RSQB — array/dict index access."""
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

    def juxt(self, meta: Meta, args: _Args) -> syntax.Expr:
        """juxt: postfix — the unaliased alternative passes the postfix expr through.

        The ``postfix juxt_arg`` alternative is aliased to :meth:`juxt_call`.
        """
        return cast(syntax.Expr, args[0])

    def juxt_call(self, meta: Meta, args: _Args) -> syntax.Call:
        """Single-arg call sugar: ``juxt: postfix juxt_arg -> juxt_call``.

        `f x` desugars to `Call(callee=f, args=(arg,), named_args=())`, where
        the second argument is the juxtaposed expression (the ``juxt_arg`` Expr).

        A ``::[T]`` suffix on the callee is folded into the call's own type
        arguments, exactly as :meth:`call` folds it for the parenthesized form:
        ``f::[int] x`` and ``f::[int](x)`` are one call spelled two ways. Left
        as a ``TypeApply`` callee, the call would instead read as a specialized
        *value* being applied, which the built-ins — callable but never values —
        have no form for.
        """
        # args[0] is the callee (postfix result); args[1] is the juxtaposed
        # expression (a juxt_arg Expr).
        callee, type_args = _split_type_apply(cast(syntax.Expr, args[0]))
        arg_expr = cast(syntax.Expr, args[1])
        return syntax.Call(
            callee=callee,
            type_args=type_args,
            args=(arg_expr,),
            named_args=(),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def juxt_arg(self, meta: Meta, args: _Args) -> syntax.Expr:
        """juxt_arg: juxt_atom juxt_arg_tail

        Builds the restricted postfix chain allowed by single-arg call sugar,
        such as ``print res.stdout``, ``print xs[0]``, and
        ``f Opt::Some(x = 1)`` and ``f Opt[int]::None()``.
        """
        base, suffixes = args
        return self._apply_juxt_suffixes(
            cast(syntax.Expr, base), cast(tuple[_JuxtSuffix, ...], suffixes), meta
        )

    def juxt_suffixes_empty(self, meta: Meta, args: _Args) -> tuple[_JuxtSuffix, ...]:
        """Represent the end of a juxtaposition postfix chain."""
        del meta, args
        return ()

    def juxt_postfix_tail(self, meta: Meta, args: _Args) -> tuple[_JuxtSuffix, ...]:
        """Prepend a non-member postfix suffix to a juxtaposition chain."""
        del meta
        suffix, tail = cast(tuple[_JuxtSuffix, tuple[_JuxtSuffix, ...]], tuple(args))
        return (suffix, *tail)

    def juxt_field_name(self, meta: Meta, args: _Args) -> _JuxtField:
        """Allocate a juxtaposition field before later postfix payloads are read."""
        del meta
        field = _find_name_token(args)
        return _JuxtField(
            name=str(field), span=token_span(field, self._source), node_id=self._next_id()
        )

    def juxt_field_tail(self, meta: Meta, args: _Args) -> tuple[_JuxtSuffix, ...]:
        """Prepend a member projection to a juxtaposition postfix chain."""
        del meta
        field = next(arg for arg in args if isinstance(arg, _JuxtField))
        tail = next(arg for arg in args if isinstance(arg, tuple))
        return (
            _JuxtSuffix("field", field.name, field.span, field.node_id),
            *cast(tuple[_JuxtSuffix, ...], tail),
        )

    def _apply_juxt_suffixes(
        self, result: syntax.Expr, suffixes: Iterable[_JuxtSuffix], meta: Meta | None = None
    ) -> syntax.Expr:
        """Apply deferred juxtaposition postfix operations to ``result``.

        A supplied *meta* spans the whole juxtaposition chain.  Omitting it
        selects source-ordered spans covering only the receiver and the applied
        suffix, which is what a raw-member receiver needs.
        """
        for suffix in suffixes:
            span = (
                self._span_from_meta(meta)
                if meta is not None
                else span_covering(result.span, suffix.span)
            )
            if suffix.kind == "field":
                result = syntax.FieldAccess(
                    obj=result,
                    field=cast(str, suffix.value),
                    span=span,
                    node_id=suffix.node_id,
                )
            elif suffix.kind == "index":
                result = syntax.IndexAccess(
                    obj=result,
                    index=cast(syntax.Expr, suffix.value),
                    span=span,
                    node_id=suffix.node_id,
                )
            else:
                type_args_val, arg_lists = cast(_JuxtCall, suffix.value)
                pos_args, named_args = arg_lists
                result = syntax.Call(
                    callee=result,
                    args=tuple(pos_args),
                    named_args=tuple(named_args),
                    type_args=type_args_val,
                    span=span,
                    node_id=suffix.node_id,
                )
        return result

    def juxt_index_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_postfix_suffix: INDEX_LSQB expr RSQB -> juxt_index_suffix."""
        index_expr = cast(syntax.Expr, next(a for a in args if _is_expr_node(a)))
        return _JuxtSuffix("index", index_expr, self._span_from_meta(meta), self._next_id())

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
        """juxt_postfix_suffix: LPAR arg_list? RPAR -> juxt_call_suffix."""
        return _JuxtSuffix(
            "call",
            ((), self._juxt_finalized_arg_lists(meta, args)),
            self._span_from_meta(meta),
            self._next_id(),
        )

    def juxt_typed_call_suffix(self, meta: Meta, args: _Args) -> _JuxtSuffix:
        """juxt_postfix_suffix: type_args LPAR arg_list? RPAR."""
        return _JuxtSuffix(
            "typed_call",
            (_find_type_args(args), self._juxt_finalized_arg_lists(meta, args)),
            self._span_from_meta(meta),
            self._next_id(),
        )

    def type_apply(self, meta: Meta, args: _Args) -> syntax.TypeApply:
        """Apply explicit type arguments to a value without calling it."""
        expr = cast(syntax.Expr, args[0])
        return syntax.TypeApply(
            expr=expr,
            type_args=_find_type_args(args),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def _finalize_call_args(
        self,
        pos_args: list[_RawPosArg],
        named_args: list[_RawNamed],
        *,
        call_span: SourceSpan,
    ) -> _CallArgs:
        placeholders: list[_RawPlaceholder] = []
        for arg in pos_args:
            if isinstance(arg, _RawPlaceholder):
                placeholders.append(arg)
        for named_arg in named_args:
            if isinstance(named_arg, _RawNamedArg) and isinstance(named_arg.value, _RawPlaceholder):
                placeholders.append(named_arg.value)

        self._validate_placeholders(placeholders, call_span=call_span)

        final_pos: list[syntax.CallArg] = []
        for arg in pos_args:
            if isinstance(arg, _RawPlaceholder):
                final_pos.append(self._build_placeholder(arg))
            else:
                final_pos.append(cast(syntax.Expr, arg))

        final_named: list[syntax.NamedArg[syntax.CallArg]] = []
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
            (digits, placeholder.span)
            for placeholder in placeholders
            if (digits := placeholder.raw_digits) is not None
        ]
        for digits, span in numbered:
            if digits == "0":
                raise AglSyntaxError("placeholder index must be positive.", span=span)
            if digits.startswith("0"):
                raise AglSyntaxError("placeholder index must not have a leading zero.", span=span)
        if bare and numbered:
            raise AglSyntaxError(
                "placeholder arguments cannot mix bare and numbered forms in one call.",
                span=numbered[0][1],
            )
        if not numbered:
            return

        seen: dict[int, SourceSpan] = {}
        for digits, span in numbered:
            index = int(digits)
            if index in seen:
                raise AglSyntaxError(f"placeholder numbered index ?{index} is repeated.", span=span)
            seen[index] = span
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
                _reject_positional_after_keyed(pos_arg.span, seen_keyed=seen_named)
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

    def named_arg(self, meta: Meta, args: _Args) -> syntax.NamedArg[syntax.Expr] | _RawNamedArg:
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

    def attr_arg_list(self, meta: Meta, args: _Args) -> _RawAttrArgLists:
        """attr_arg_list: attr_arg (COMMA attr_arg)* COMMA?

        Returns (pos_args, keyed_args) for the attribute builder. Keys are
        not checked for duplicates here; each attribute's own semantics
        decide. Positional arguments after a keyed one are still rejected
        here, as for an ordinary call.
        """
        pos_args: list[_RawAttrPosArg] = []
        keyed_args: list[syntax.AttributeKeyedArg] = []
        seen_keyed = False
        for a in args:
            if isinstance(a, syntax.AttributeKeyedArg):
                keyed_args.append(a)
                seen_keyed = True
            elif _is_expr_node(a):
                pos_arg = cast(_RawAttrPosArg, a)
                _reject_positional_after_keyed(pos_arg.span, seen_keyed=seen_keyed)
                pos_args.append(pos_arg)
        return (pos_args, keyed_args)

    def attr_keyed_arg(self, meta: Meta, args: _Args) -> syntax.AttributeKeyedArg:
        """attr_keyed_arg: attr_key EQ arg_expr — a bare or qualified attribute key.

        ``attr_key`` reuses ``qual_var_ref``/``var_ref``, so the key already
        arrives as a finished ``VarRef`` — the same construction an ordinary
        reference expression gets, just picked out of the child list here.
        The value is ``arg_expr`` only: an attribute keyed argument never
        admits a placeholder. A type-applied qualifier segment
        (``Foo[int]::x``) is rejected: an attribute key names a binding, not
        a generic instantiation. See ``AttributeKeyedArg`` for the key's
        legal qualifier spellings.
        """
        key = next(a for a in args if isinstance(a, syntax.VarRef))
        if key.qualifier is not None and any(
            segment.type_args is not None for segment in key.qualifier.segments
        ):
            raise AglSyntaxError(
                "an attribute key names a binding; type arguments have no meaning here.",
                span=key.span,
            )
        # A raw infix chain is grouped later, by scope resolution.
        value = cast(syntax.Expr, next(a for a in args if _is_expr_node(a) and a is not key))
        return syntax.AttributeKeyedArg(
            key=key,
            value=value,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Record update
    # ------------------------------------------------------------------

    def with_expr(self, meta: Meta, args: _Args) -> syntax.RecordUpdate:
        """with_expr: with_target "with" with_update (COMMA with_update)*"""
        target = _find_expr(args)
        updates = tuple(a for a in args if isinstance(a, syntax.NamedArg))
        return syntax.RecordUpdate(
            target=target,
            updates=updates,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def with_update(self, meta: Meta, args: _Args) -> syntax.NamedArg[syntax.Expr]:
        """with_update: field_name EQ or_expr"""
        name_tok = _find_name_token(args)
        value = _find_expr(args[1:])
        return syntax.NamedArg(
            name=str(name_tok),
            value=value,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Binary operators
    # ------------------------------------------------------------------

    def _op(self, meta: Meta, args: _Args, name: str) -> _InfixOperator:
        return _InfixOperator(
            name=name,
            callee_node_id=self._next_id(),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def op_or(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "or")

    def op_and(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "and")

    def op_in(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "in")

    def op_eq(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "==")

    def op_neq(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "!=")

    def op_lt(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "<")

    def op_le(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "<=")

    def op_gt(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, ">")

    def op_ge(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, ">=")

    def op_add(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "+")

    def op_sub(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "-")

    def op_mul(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "*")

    def op_div(self, meta: Meta, args: _Args) -> _InfixOperator:
        return self._op(meta, args, "/")

    def op_user(self, meta: Meta, args: _Args) -> _InfixOperator:
        tok = next(a for a in args if isinstance(a, Token))
        return self._op(meta, args, str(tok))

    def not_prefix(self, meta: Meta, args: _Args) -> syntax.RawPrefixNot:
        return syntax.RawPrefixNot(span=self._span_from_meta(meta), node_id=self._next_id())

    def infix_operand(self, meta: Meta, args: _Args) -> _InfixOperand:
        expr = cast(syntax.Expr | _RawInfixChain, next(a for a in args if _is_expr_node(a)))
        return _InfixOperand(
            expr=expr,
            prefix_nots=tuple(a for a in args if isinstance(a, syntax.RawPrefixNot)),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def infix_chain(self, meta: Meta, args: _Args) -> syntax.Expr | _RawInfixChain:
        operands = tuple(a for a in args if isinstance(a, _InfixOperand))
        operators = tuple(a for a in args if isinstance(a, _InfixOperator))
        if not operators and not operands[0].prefix_nots:
            return operands[0].expr
        if groups_at_parse(operands, operators):
            return group_infix(
                operands, operators, [BUILTIN_FIXITIES[operator.name] for operator in operators]
            )
        return _RawInfixChain(
            operands=operands,
            operators=operators,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
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

    def _make_is_test(self, meta: Meta, args: _Args, *, negated: bool) -> syntax.IsTest:
        """cast "is" "not"? (qual_ref_chain | name) — a qualified or simple variant test."""
        target = args[-1]
        qualifier = target if isinstance(target, syntax.QualifierChain) else None
        return syntax.IsTest(
            expr=cast(syntax.Expr, args[0]),
            variant=str(target) if qualifier is None else qualifier.member,
            negated=negated,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            qualifier=qualifier,
        )

    def is_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, negated=False)

    def is_not_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        return self._make_is_test(meta, args, negated=True)

    is_test_qualified = is_test_simple
    is_not_test_qualified = is_not_test_simple

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
        pat, _arrow, body = args
        return syntax.CaseBranch(
            pattern=cast(syntax.Pattern, pat),
            body=cast(syntax.Expr, body),
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

    # range_tail / range_dir / range_step transformers.
    # These produce intermediate tuples consumed by for_clause.

    def range_to(self, meta: Meta, args: _Args) -> bool:
        """range_dir: TO -> range_to — direction flag False (ascending)."""
        return False

    def range_downto(self, meta: Meta, args: _Args) -> bool:
        """range_dir: DOWNTO -> range_downto — direction flag True (descending)."""
        return True

    def range_step(self, meta: Meta, args: _Args) -> syntax.Expr:
        """range_step: STEP or_expr — return the step expression."""
        return cast(syntax.Expr, _find_non_token(args))

    def range_tail(self, meta: Meta, args: _Args) -> tuple[bool, syntax.Expr, syntax.Expr | None]:
        """range_tail: range_dir or_expr range_step?

        Returns (is_downto, to_bound_expr, by_step_expr_or_None).
        """
        is_down, to_bound, *step = args
        by_step = cast(syntax.Expr, step[0]) if step else None
        return (cast(bool, is_down), cast(syntax.Expr, to_bound), by_step)

    def for_clause(
        self, meta: Meta, args: _Args
    ) -> tuple[str, syntax.Expr, syntax.Expr | None, bool, syntax.Expr | None]:
        """for_clause: "for" name "in" or_expr range_tail? _NEWLINE?

        Returns a 5-tuple:
          (var_name, start_expr, range_to_expr, range_down, range_step_expr)

        For a collection for (no range_tail):
          range_to_expr=None, range_down=False, range_step_expr=None.
        For a range for (range_tail present):
          range_to_expr is the upper/lower bound; range_down is True for downto;
          range_step_expr is the step or None for default step.
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
            range_down, range_to, range_step = range_tail_result
        else:
            range_down, range_to, range_step = False, None, None
        return (str(name_tok), start_expr, range_to, range_down, range_step)

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
        """inline_seq: (marked_item SEMICOLON)* marked_item."""
        return self._block_from_items(meta, args)

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
        for_range_step: syntax.Expr | None = None
        if clauses[0] is not None:
            for_var, for_iter, for_range_to, for_range_down, for_range_step = clauses[0]
        while_cond: syntax.Expr | None = clauses[1]

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
            for_range_step=for_range_step,
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
        """try_body: suite_expr | (marked_item SEMICOLON)* try_tail.

        Returns a Block for a sequence, or the single item on its own.
        """
        items = [a for a in args if a is not None and not isinstance(a, Token)]
        if len(items) == 1:
            return self._body_as_expr(meta, cast(_RawItem, items[0]))
        return self._block_from_items(meta, args)

    def catch_pattern(self, meta: Meta, args: _Args) -> tuple[str | None, str | None]:
        """catch_pattern: name ("as" name)?

        Handles any NAME. Wildcard is "_" (NAME).
        """
        first, *binding = (str(a) for a in args)
        return (None if first == "_" else first, binding[0] if binding else None)

    def catch_body(self, meta: Meta, args: _Args) -> syntax.Expr:
        """catch_body: suite_expr | closed_item."""
        return self._body_as_expr(meta, cast(_RawItem, _find_non_token(args)))

    def catch_clause(self, meta: Meta, args: _Args) -> syntax.CatchClause:
        """catch_clause: "catch" catch_pattern ARROW catch_body"""
        pattern, _arrow, body = args
        exc_type, binding = cast(tuple[str | None, str | None], pattern)
        return syntax.CatchClause(
            exc_type=exc_type,
            binding=binding,
            body=cast(syntax.Expr, body),
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
        """branch_body: suite_expr | closed_item."""
        return self._body_as_expr(meta, cast(_RawItem, _find_non_token(args)))

    def inline_assign(self, meta: Meta, args: _Args) -> syntax.AssignStmt:
        """inline_assign: postfix ASSIGN or_expr.

        Same node as assign_stmt; the grammar restricts only the RHS.
        """
        return self.assign_stmt(meta, args)

    def paren_block(self, meta: Meta, args: _Args) -> syntax.Block:
        """paren_block: marked sequence, inline assignment, or lone binder."""
        return self._block_from_items(meta, args)

    def paren_block_expr(self, meta: Meta, args: _Args) -> syntax.Block:
        """paren_expr_or_unit: LPAR paren_block RPAR — unwrap to the Block."""
        return next(a for a in args if isinstance(a, syntax.Block))

    def operator_ref(self, meta: Meta, args: _Args) -> syntax.OperatorRef:
        """paren_expr_or_unit: LPAR operator_ref_op RPAR — a built-in operator value."""
        return syntax.OperatorRef(
            op=BUILTIN_OPS[str(args[1])],
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def pat_var_or_wild(
        self, meta: Meta, args: _Args
    ) -> syntax.WildcardPattern | syntax.VarPattern:
        """name → WildcardPattern (when value is "_") or VarPattern."""
        tok = cast(Token, args[0])
        if str(tok) == "_":
            return syntax.WildcardPattern(span=self._span_from_meta(meta), node_id=self._next_id())
        return syntax.VarPattern(
            name=str(tok), span=self._span_from_meta(meta), node_id=self._next_id()
        )

    def pat_as(self, meta: Meta, args: _Args) -> syntax.Pattern:
        """pattern: pattern_atom ("as" name)* — build left-associated binders."""
        pattern = cast(syntax.Pattern, args[0])
        for name_tok in args[1:]:
            name = str(name_tok)
            if name == "_":
                raise AglSyntaxError(
                    "An as-pattern binder cannot be '_'.", span=self._span_from_meta(meta)
                )
            pattern = syntax.AsPattern(
                pattern=pattern,
                name=name,
                span=self._span_from_meta(meta),
                node_id=self._next_id(),
            )
        return pattern

    def pat_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        """pat_constructor: name LPAR pattern_fields? RPAR"""
        name = str(args[0])
        positional, named = _pattern_fields(args)
        return syntax.ConstructorPattern(
            qualifier=None,
            name=name,
            positional=positional,
            named=named,
            span=self._span_from_meta(meta),
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
        tok = cast(Token, args[0])
        lit = syntax.IntLit(
            value=int(str(tok)), span=self._span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_decimal(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = cast(Token, args[0])
        lit = self._build_decimal_lit(tok, meta)
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
            cast(syntax.StringLit | syntax.Template, args[0]),
            "Pattern string literals cannot contain interpolation.",
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
        """pat_field_named: field_name EQ pattern"""
        return syntax.PatternField(
            name=str(args[0]),
            pattern=cast(syntax.Pattern, args[-1]),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def pat_field_positional(self, meta: Meta, args: _Args) -> syntax.Pattern:
        """pat_field_positional: pattern — return the sub-pattern directly."""
        return cast(syntax.Pattern, args[0])

    # ------------------------------------------------------------------
    # Module declarations
    # ------------------------------------------------------------------

    def import_alias(self, meta: Meta, args: _Args) -> str:
        """Return a module alias name."""
        return str(next(a for a in args if _is_name_token(a)))

    def use_alias(self, meta: Meta, args: _Args) -> str:
        """Return an additive route alias name."""
        return str(next(a for a in args if _is_name_token(a)))

    def selection_atom(self, meta: Meta, args: _Args) -> _SelectedAtom:
        """Keep one selected path atom until its declaration chooses its AST type."""
        path = next(a for a in args if isinstance(a, _ScopePath))
        rename = next((str(a) for a in args if _is_name_token(a)), None)
        return _SelectedAtom(path=path, rename=rename, span=self._span_from_meta(meta))

    def brace_atom(self, meta: Meta, args: _Args) -> _SelectedAtom:
        """Pass through a brace atom."""
        return next(a for a in args if isinstance(a, _SelectedAtom))

    def brace_glob(self, meta: Meta, args: _Args) -> Token:
        """Keep an invalid brace glob for declaration-level validation."""
        return _required_token(args, "STAR")

    def brace_nested(self, meta: Meta, args: _Args) -> _Braces:
        """Keep an invalid nested group for declaration-level validation."""
        return next(a for a in args if isinstance(a, _Braces))

    def brace_entries(self, meta: Meta, args: _Args) -> tuple[_SelectedAtom | _Braces | Token, ...]:
        """Collect braces entries without validating their source form yet."""
        return tuple(
            a for a in args if isinstance(a, (_SelectedAtom, _Braces)) or _is_typed_token(a, "STAR")
        )

    def braces(self, meta: Meta, args: _Args) -> _Braces:
        """Build a brace group, including an empty one for a targeted error."""
        entries = next((a for a in args if isinstance(a, tuple)), ())
        return _Braces(entries=cast(tuple[_SelectedAtom | _Braces | Token, ...], entries))

    def selection_glob(self, meta: Meta, args: _Args) -> _Selection:
        """Build a glob selection."""
        return _Selection()

    def selection_braces(self, meta: Meta, args: _Args) -> _Selection:
        """Build a brace selection."""
        return _Selection(braces=next(a for a in args if isinstance(a, _Braces)))

    def selection_single(self, meta: Meta, args: _Args) -> _Selection:
        """Build a single-atom selection."""
        return _Selection(atom=next(a for a in args if isinstance(a, _SelectedAtom)))

    def import_tail_clause(self, meta: Meta, args: _Args) -> _Selection:
        """Pass through an import tail selection."""
        return next(a for a in args if isinstance(a, _Selection))

    def export_braces_tail(self, meta: Meta, args: _Args) -> _Braces:
        """Pass through export braces."""
        return next(a for a in args if isinstance(a, _Braces))

    def export_glob_tail(self, meta: Meta, args: _Args) -> Token:
        """Keep an invalid export glob for declaration-level validation."""
        return _required_token(args, "STAR")

    def hiding_clause(self, meta: Meta, args: _Args) -> tuple[_ScopePath, ...]:
        """Collect the paths hidden by one declaration."""
        return tuple(a for a in args if isinstance(a, _ScopePath))

    def _trim_token_span(self, token: Token, *, start: int = 0, end: int = 0) -> SourceSpan:
        """Return a token span without synthetic use-target delimiters."""
        span = token_span(token, self._source)
        return SourceSpan(
            start_line=span.start_line,
            start_col=span.start_col + start,
            end_line=span.end_line,
            end_col=span.end_col - end,
            start_offset=span.start_offset + start,
            end_offset=span.end_offset - end,
            source=span.source,
        )

    def _scope_segment(self, token: Token) -> tuple[str, SourceSpan]:
        """Return the scope segment one merged qualifier token spells.

        A qualifier written inside a path names a scope, so a module route
        there — a separator anywhere in the token, the leading anchor
        included — is rejected rather than carried into a name that could
        never resolve.
        """
        if _ROUTE_SEPARATOR in str(token):
            raise AglSyntaxError(_MODULE_ROUTE_MESSAGE, span=token_span(token, self._source))
        route = self._route_token(token)
        return route.name, route.span

    def _route_token(self, token: Token) -> _RouteToken:
        """Split a merged ``MODPATH`` or ``MODQUAL`` token into its parts.

        A merged token keeps the ``/`` anchoring it in its value, and a
        qualifier's span still covers the ``::`` joining it to what follows
        though its value does not. Accounting for both in one place is what
        lets every builder underline the name and nothing else.
        """
        spelling = str(token)
        anchored = spelling.startswith(_ROUTE_SEPARATOR)
        delimited = _is_typed_token(token, "MODQUAL")
        return _RouteToken(
            name=spelling.removeprefix(_ROUTE_SEPARATOR),
            anchored=anchored,
            span=self._trim_token_span(
                token,
                start=len(_ROUTE_SEPARATOR) if anchored else 0,
                end=len(_QUALIFIER_DELIMITER) if delimited else 0,
            ),
        )

    def _path_span(self, path: _ScopePath) -> SourceSpan:
        """Return the span from the first through last segment of a path."""
        return span_covering(path.segments[0][1], path.segments[-1][1])

    def _selection_item(
        self, cls: type[_SelectionItemT], path: _ScopePath, rename: str | None, span: SourceSpan
    ) -> _SelectionItemT:
        """Build one import or export selection item from a parsed path."""
        name, _name_span = path.segments[-1]
        return cls(
            name=name,
            rename=rename,
            scope_path=self._scope_segments(_ScopePath(path.segments[:-1])),
            span=span,
            node_id=self._next_id(),
        )

    def _brace_atoms(self, braces: _Braces, span: SourceSpan) -> tuple[_SelectedAtom, ...]:
        """Validate one brace group and return its direct atom entries."""
        if not braces.entries:
            raise AglSyntaxError("Selection braces cannot be empty.", span=span)
        if any(isinstance(entry, Token) for entry in braces.entries):
            raise AglSyntaxError("Selection braces cannot contain '*'.", span=span)
        if any(isinstance(entry, _Braces) for entry in braces.entries):
            raise AglSyntaxError("Selection braces cannot be nested.", span=span)
        return cast(tuple[_SelectedAtom, ...], braces.entries)

    def _selection_atoms(
        self, selection: _Selection, span: SourceSpan
    ) -> tuple[_SelectedAtom, ...]:
        """Return normalized tail atoms, using an empty tuple for a glob."""
        if selection.braces is not None:
            return self._brace_atoms(selection.braces, span)
        if selection.atom is not None:
            return (selection.atom,)
        return ()

    def import_decl(self, meta: Meta, args: _Args) -> syntax.ImportDecl:
        """Build an import declaration and enforce its tail and hiding rules."""
        span = self._span_from_meta(meta)
        module_path = _module_path(args)
        wildcard = _typed_token(args, "WILDCARD") is not None
        alias = next((a for a in args if type(a) is str), None)
        tail = next((a for a in args if isinstance(a, _Selection)), None)
        hidden_paths = next((a for a in args if isinstance(a, tuple)), ())
        hidden_paths = cast(tuple[_ScopePath, ...], hidden_paths)
        if alias is not None and tail is not None:
            raise AglSyntaxError("An import alias cannot be combined with a tail.", span=span)
        if hidden_paths and tail is not None and not tail.glob:
            raise AglSyntaxError("Hiding is only valid with a plain or glob import.", span=span)
        tail_atoms = () if tail is None else self._selection_atoms(tail, span)
        return syntax.ImportDecl(
            module_path=module_path,
            wildcard=wildcard,
            alias=alias,
            tail=None
            if tail is None
            else tuple(
                self._selection_item(syntax.ImportItem, atom.path, atom.rename, atom.span)
                for atom in tail_atoms
            ),
            hidden=tuple(
                self._selection_item(syntax.ImportItem, path, None, self._path_span(path))
                for path in hidden_paths
            ),
            span=span,
            node_id=self._next_id(),
        )

    def use_suffix_alias(self, meta: Meta, args: _Args) -> _UseTarget:
        """Collect a whole-target alias without committing its target spelling."""
        return (), None, next(a for a in args if type(a) is str), None

    def use_suffix(self, meta: Meta, args: _Args) -> _UseTarget:
        """Pass through one colon-introduced unresolved use target."""
        return next(a for a in args if isinstance(a, tuple))

    def use_anchored_suffix(self, meta: Meta, args: _Args) -> _UseTarget:
        """Pass through one current-module unresolved use target."""
        return next(a for a in args if isinstance(a, tuple))

    def use_path_alias(self, meta: Meta, args: _Args) -> _UseTarget:
        """Collect an unresolved named use target with a trailing alias."""
        return (
            _typed_tokens(args, "MODQUAL"),
            _required_token(args, "NAME"),
            next(a for a in args if type(a) is str),
            None,
        )

    def use_path_single(self, meta: Meta, args: _Args) -> _UseTarget:
        """Build a single-member use tail after its unresolved target path."""
        prefixes = _typed_tokens(args, "MODQUAL")
        target = _required_token(args, "NAME")
        atom = _SelectedAtom(
            path=_ScopePath(((str(target), token_span(target, self._source)),)),
            rename=None,
            span=token_span(target, self._source),
        )
        return prefixes, None, None, _Selection(atom=atom)

    def use_path_glob(self, meta: Meta, args: _Args) -> _UseTarget:
        """Build a glob use tail after its unresolved target path."""
        return (
            _typed_tokens(args, "MODQUAL"),
            None,
            None,
            _Selection(),
        )

    def use_path_braces(self, meta: Meta, args: _Args) -> _UseTarget:
        """Build a braced use tail after its unresolved target path."""
        braces = next(a for a in args if isinstance(a, _Braces))
        return (
            _typed_tokens(args, "MODQUAL"),
            None,
            None,
            _Selection(braces=braces),
        )

    def use_decl(self, meta: Meta, args: _Args) -> syntax.UseDecl:
        """Build a use declaration and enforce its injection and hiding rules."""
        span = self._span_from_meta(meta)
        path_token = _typed_token(args, "MODPATH")
        current_module = path_token is None
        anchored = current_module
        target_segments: list[tuple[str, SourceSpan]] = []
        if path_token is not None:
            route = self._route_token(path_token)
            anchored = route.anchored
            target_segments.append((route.name, route.span))
        parts = cast(
            _UseTarget,
            next(
                a
                for a in args
                if isinstance(a, tuple)
                and len(a) == 4
                and isinstance(a[0], tuple)
                and (not a[0] or isinstance(a[0][0], Token))
            ),
        )
        prefixes, final_target, alias, tail = parts
        for token in prefixes:
            target_segments.append(self._scope_segment(token))
        if final_target is not None:
            target_segments.append((str(final_target), token_span(final_target, self._source)))
        hidden_paths = cast(
            tuple[_ScopePath, ...],
            next(
                (
                    a
                    for a in args
                    if isinstance(a, tuple) and all(isinstance(path, _ScopePath) for path in a)
                ),
                (),
            ),
        )
        if hidden_paths and (tail is None or not tail.glob):
            raise AglSyntaxError("Hiding is only valid with a glob use tail.", span=span)
        tail_atoms = () if tail is None else self._selection_atoms(tail, span)
        return syntax.UseDecl(
            anchored=anchored,
            target=self._scope_segments(_ScopePath(tuple(target_segments))),
            tail=None
            if tail is None
            else tuple(
                self._selection_item(syntax.ImportItem, atom.path, atom.rename, atom.span)
                for atom in tail_atoms
            ),
            hidden=tuple(
                self._selection_item(syntax.ImportItem, path, None, self._path_span(path))
                for path in hidden_paths
            ),
            alias=alias,
            span=span,
            node_id=self._next_id(),
            current_module=current_module,
        )

    def export_decl(self, meta: Meta, args: _Args) -> syntax.ExportDecl:
        """Build an export declaration and enforce its braces and hiding rules."""
        span = self._span_from_meta(meta)
        module_path = _module_path(args)
        wildcard = _typed_token(args, "WILDCARD") is not None
        if _typed_token(args, "STAR") is not None:
            raise AglSyntaxError("Export declarations cannot use a glob tail.", span=span)
        braces = next((a for a in args if isinstance(a, _Braces)), None)
        hidden_paths = cast(
            tuple[_ScopePath, ...], next((a for a in args if isinstance(a, tuple)), ())
        )
        if braces is not None and hidden_paths:
            raise AglSyntaxError("Export braces cannot be combined with hiding.", span=span)
        atoms = () if braces is None else self._brace_atoms(braces, span)
        return syntax.ExportDecl(
            module_path=module_path,
            wildcard=wildcard,
            items=tuple(
                self._selection_item(syntax.ExportItem, atom.path, atom.rename, atom.span)
                for atom in atoms
            ),
            hidden=tuple(
                self._selection_item(syntax.ExportItem, path, None, self._path_span(path))
                for path in hidden_paths
            ),
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # Builtin declarations
    # ------------------------------------------------------------------

    def builtin_record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        """builtin_record_def: attributes? BUILTIN _record_def_tail"""
        return replace(self.record_def(meta, args), is_builtin=True)

    def builtin_enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        """builtin_enum_def: attributes? BUILTIN _enum_def_tail"""
        return replace(self.enum_def(meta, args), is_builtin=True)

    def builtin_exception_def(self, meta: Meta, args: _Args) -> syntax.ExceptionDef:
        """builtin_exception_def: attributes? BUILTIN _exception_def_tail"""
        return replace(self.exception_def(meta, args), is_builtin=True)

    def builtin_type_alias(self, meta: Meta, args: _Args) -> syntax.TypeAlias:
        """builtin_type_alias: attributes? BUILTIN _type_alias_tail"""
        return replace(self.type_alias(meta, args), is_builtin=True)

    # ------------------------------------------------------------------
    # Qualified refs
    # ------------------------------------------------------------------

    def qualifier_mod_segment(self, meta: Meta, args: _Args) -> _QualifierChainSegment:
        """Build one chain segment from a lexer-merged qualifier token."""
        token = cast(Token, args[0])
        route = self._route_token(token)
        return _QualifierChainSegment(
            segment=syntax.QualifierSegment(
                name=route.name,
                type_args=None,
                span=route.span,
                node_id=self._next_id(),
                anchored=route.anchored,
            ),
            anchored=route.anchored,
        )

    def qualifier_applied_segment(self, meta: Meta, args: _Args) -> _QualifierChainSegment:
        """Build one type-applied qualifier-chain segment."""
        name_tok = next(arg for arg in args if _is_name_token(arg))
        type_args = cast(
            tuple[TypeExpr, ...],
            next(
                arg
                for arg in args
                if isinstance(arg, tuple) and (not arg or isinstance(arg[0], _ALL_TYPE_EXPRS))
            ),
        )
        end_token = next(
            arg for arg in reversed(args) if isinstance(arg, Token) and arg.type == "RSQB"
        )
        type_start = token_span(name_tok, self._source)
        type_end = token_span(end_token, self._source)
        return _QualifierChainSegment(
            segment=syntax.QualifierSegment(
                name=str(name_tok),
                type_args=type_args,
                span=span_covering(type_start, type_end),
                node_id=self._next_id(),
            )
        )

    def _expression_chain(
        self,
        anchor: syntax.QualifierAnchor | None,
        segments: list[_QualifierChainSegment],
        member: Token,
        span: SourceSpan,
    ) -> syntax.QualifierChain:
        """Build a chain from its parsed segments and optional anchor."""
        return syntax.QualifierChain(
            anchor=anchor,
            segments=tuple(segment.segment for segment in segments),
            member=str(member),
            span=span,
            node_id=self._next_id(),
        )

    def qual_ref_chain(self, meta: Meta, args: _Args) -> syntax.QualifierChain:
        """Build a qualified-reference chain for every qualified position."""
        name_tok = next(arg for arg in reversed(args) if _is_name_token(arg))
        segments = [arg for arg in args if isinstance(arg, _QualifierChainSegment)]
        first = args[0]
        anchor = (
            syntax.QualifierAnchor.CURRENT_MODULE
            if isinstance(first, Token) and first.type == "DCOLON"
            else (
                syntax.QualifierAnchor.MODULE
                if isinstance(first, _QualifierChainSegment) and first.anchored
                else None
            )
        )
        return self._expression_chain(anchor, segments, name_tok, self._span_from_meta(meta))

    def qual_var_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        """qual_var_ref: qual_ref_chain"""
        chain = cast(syntax.QualifierChain, args[0])
        return syntax.VarRef(
            name=chain.member, span=chain.span, node_id=self._next_id(), qualifier=chain
        )

    def qual_named_type(self, meta: Meta, args: _Args) -> NameT:
        """qual_ref_chain in type position."""
        qualifier = cast(syntax.QualifierChain, args[0])
        return NameT(
            name=qualifier.member,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            qualifier=qualifier,
        )

    def pat_qual_bare(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        """pat_qual_bare: qual_ref_chain — a qualified pattern written without parens."""
        qualifier = next(a for a in args if isinstance(a, syntax.QualifierChain))
        node_id = self._next_id()
        return syntax.ConstructorPattern(
            name=qualifier.member,
            positional=(),
            named=(),
            span=self._span_from_meta(meta),
            node_id=node_id,
            qualifier=qualifier,
        )

    def pat_qual_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        """pat_qual_constructor: qual_ref_chain LPAR pattern_fields? RPAR"""
        qualifier = next(a for a in args if isinstance(a, syntax.QualifierChain))
        positional, named = _pattern_fields(args)
        return syntax.ConstructorPattern(
            name=qualifier.member,
            positional=positional,
            named=named,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
            qualifier=qualifier,
        )

    # ------------------------------------------------------------------
    # template desugaring
    # ------------------------------------------------------------------

    def template(self, meta: Meta, args: _Args) -> syntax.Template | syntax.StringLit:
        """Build a quoted or `$` verbatim template expression from its segment children.

        Non-segment children (delimiter tokens) are skipped, so quoted
        templates and `$` verbatim literals can hand their raw argument list
        straight through; a hole-free result collapses to plain text.
        """
        nonempty_segments = tuple(
            segment
            for segment in args
            if isinstance(segment, (syntax.TextSegment, syntax.InterpSegment))
            and not (isinstance(segment, syntax.TextSegment) and segment.text == "")
        )
        span = self._span_from_meta(meta)
        nid = self._next_id()
        if all(isinstance(segment, syntax.TextSegment) for segment in nonempty_segments):
            text = "".join(
                segment.text
                for segment in nonempty_segments
                if isinstance(segment, syntax.TextSegment)
            )
            return syntax.StringLit(value=text, span=span, node_id=nid)
        return syntax.Template(segments=nonempty_segments, span=span, node_id=nid)

    def type_args(self, meta: Meta, args: _Args) -> tuple[TypeExpr, ...]:
        """Pass an explicit ``::[...]`` type-argument group through to its call."""
        del meta
        return _find_type_args(args)

    def tmpl_text(self, meta: Meta, args: _Args) -> syntax.TextSegment:
        tok = cast(Token, args[0])
        return syntax.TextSegment(
            text=str(tok),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def tmpl_interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        return cast(syntax.InterpSegment, args[0])

    def interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        # Grammar: INTERP_START expr INTERP_END
        expr: syntax.Expr = _find_expr(args)
        return syntax.InterpSegment(
            expr=expr,
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )

    def tmpl_env(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        """``${NAME}`` — an environment hole, a call reading the environment.

        Every node built here spans the hole the author wrote, so the hole
        resolves, types, and runs exactly like the qualified call spelled out
        in its place, and its diagnostics stay on the hole.
        """
        del meta
        token = cast(Token, args[0])
        span = token_span(token, self._source)
        qualifier = syntax.QualifierChain(
            anchor=None,
            segments=(
                syntax.QualifierSegment(
                    name=_ENVIRONMENT_MODULE,
                    type_args=None,
                    span=span,
                    node_id=self._next_id(),
                ),
            ),
            member=_ENVIRONMENT_READER,
            span=span,
            node_id=self._next_id(),
        )
        call = syntax.Call(
            callee=syntax.VarRef(
                name=_ENVIRONMENT_READER,
                span=span,
                node_id=self._next_id(),
                qualifier=qualifier,
            ),
            args=(syntax.StringLit(value=str(token), span=span, node_id=self._next_id()),),
            named_args=(),
            span=span,
            node_id=self._next_id(),
        )
        return syntax.InterpSegment(expr=call, span=span, node_id=self._next_id())

    # ------------------------------------------------------------------
    # Array and dict literals
    # ------------------------------------------------------------------

    def lit_array(self, meta: Meta, args: _Args) -> syntax.ArrayLit:
        """lit_array: LSQB (expr (COMMA expr)* COMMA?)? RSQB"""
        elements = tuple(
            cast(syntax.Expr, a) for a in args if a is not None and not isinstance(a, Token)
        )
        return syntax.ArrayLit(
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

    def dict_entry(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: arg_expr COLON arg_expr — key and value are both ordinary expressions."""
        key, _colon, value = args
        return syntax.DictEntry(
            key=cast(syntax.Expr, key),
            value=cast(syntax.Expr, value),
            span=self._span_from_meta(meta),
            node_id=self._next_id(),
        )


# ---------------------------------------------------------------------------
# Helper: find first TypeExpr in an args list
# ---------------------------------------------------------------------------


def _find_type_expr(args: _Args) -> TypeExpr:
    """Return the first element that is a TypeExpr instance."""
    return next(a for a in args if isinstance(a, _ALL_TYPE_EXPRS))


def _find_type_args(args: _Args) -> tuple[TypeExpr, ...]:
    """Return the explicit type-argument group among *args*, or ``()`` when absent.

    A ``type_arg_list`` result is the sole nonempty tuple of TypeExprs the
    grammar produces, so every rule with an optional ``::[...]`` group — applied
    types, typed calls, and ``type_apply`` — locates it the same way here
    rather than re-spelling the predicate inline.
    """
    for a in args:
        if (
            isinstance(a, tuple)
            and len(a) > 0
            and all(isinstance(item, _ALL_TYPE_EXPRS) for item in a)
        ):
            return cast(tuple[TypeExpr, ...], a)
    return ()


def _find_name_token(args: _Args) -> Token:
    """Return the first ``name``/``field_name`` Token (``NAME`` or ``OP_NAME``) in *args*."""
    return next(a for a in args if _is_name_token(a))


def _is_name_token(value: object) -> TypeGuard[Token]:
    return isinstance(value, Token) and value.type in _NAME_TOKEN_TYPES


def _is_typed_token(value: object, token_type: str) -> TypeGuard[Token]:
    """True iff *value* is a lexer token of *token_type*."""
    return isinstance(value, Token) and value.type == token_type


def _typed_tokens(args: _Args, token_type: str) -> tuple[Token, ...]:
    """Return every *token_type* token among a rule's children, in order."""
    return tuple(a for a in args if _is_typed_token(a, token_type))


def _typed_token(args: _Args, token_type: str) -> Token | None:
    """Return the first *token_type* token among a rule's children, if any."""
    tokens = _typed_tokens(args, token_type)
    return tokens[0] if tokens else None


def _required_token(args: _Args, token_type: str) -> Token:
    """Return the *token_type* token the grammar guarantees is among *args*."""
    return _typed_tokens(args, token_type)[0]


def _module_path(args: _Args) -> tuple[str, ...]:
    """Return the segments of the merged module path a header carries."""
    return tuple(str(_required_token(args, "MODPATH")).split(_ROUTE_SEPARATOR))


def _is_field_tuple(a: object) -> bool:
    """True iff *a* is a field- or parameter-list result (``field_list``/``param_list``).

    An empty tuple is treated as a field tuple (the ``field_list?`` absent case).
    """
    return isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], syntax.Param))


def _is_attribute_tuple(a: object) -> bool:
    """True iff *a* is an ``attributes`` result (always at least one attribute)."""
    return isinstance(a, tuple) and len(a) > 0 and isinstance(a[0], syntax.Attribute)


def _split_type_apply(callee: syntax.Expr) -> tuple[syntax.Expr, tuple[TypeExpr, ...]]:
    """Split a call's callee from the type arguments a ``TypeApply`` supplies it."""
    if isinstance(callee, syntax.TypeApply):
        return callee.expr, callee.type_args
    return callee, ()


def _pattern_fields(
    args: _Args,
) -> tuple[tuple[syntax.Pattern, ...], tuple[syntax.PatternField, ...]]:
    """Return a constructor pattern's positional and named fields, empty if it has none."""
    split = next((a for a in args if isinstance(a, _PatternFieldsSplit)), None)
    return (split.positional, split.named) if split is not None else ((), ())


def _find_type_params(args: _Args) -> tuple[str, ...]:
    """Return the type-parameter slots among a declaration's children, or none."""
    slots: tuple[str, ...] = ()
    for a in args:
        if _is_str_tuple(a):
            slots = cast(tuple[str, ...], a)
    return slots


def _is_constraint_tuple(a: object) -> bool:
    """True iff *a* is a ``constraint_block`` result (always at least one constraint)."""
    return isinstance(a, tuple) and len(a) > 0 and isinstance(a[0], syntax.Constraint)


def _find_constraints(args: _Args) -> tuple[syntax.Constraint, ...]:
    """Return a declaration's constraint block among its children, or none."""
    return next(
        (cast(tuple[syntax.Constraint, ...], a) for a in args if _is_constraint_tuple(a)), ()
    )


def _find_attributes(args: _Args) -> tuple[syntax.Attribute, ...]:
    """Return the attribute prefix among a rule's children, or none."""
    return next(
        (cast(tuple[syntax.Attribute, ...], a) for a in args if _is_attribute_tuple(a)),
        (),
    )


def _without_attributes(args: _Args) -> _Args:
    """Return a rule's children with its attribute prefix removed.

    Callers that index into their children (``args[1:]`` to skip a leading
    keyword or name) work on this list, so an optional attribute prefix cannot
    shift the positions they rely on.
    """
    return [a for a in args if not _is_attribute_tuple(a)]


def _find_field_tuple(args: _Args) -> tuple[syntax.Param, ...]:
    """Return the declaration's field tuple; a body-less declaration has none."""
    return next((cast(tuple[syntax.Param, ...], a) for a in args if _is_field_tuple(a)), ())


def _check_function_param_annotations(entries: tuple[syntax.Param, ...]) -> None:
    """Require annotations on named-function parameters except a leading ``self``."""
    for index, entry in enumerate(entries):
        if entry.type_expr is not None or (index == 0 and entry.name == _SELF_PARAM):
            continue
        message = f"Parameter {entry.name!r} has no type annotation."
        if entry.name == _SELF_PARAM:
            message += f" A bare {_SELF_PARAM!r} must be the first parameter."
        raise AglSyntaxError(message, span=entry.span)


def _declares_method(entries: tuple[syntax.Param, ...]) -> bool:
    """Whether a leading ``self`` receiver makes the declaration a method."""
    return bool(entries) and entries[0].name == _SELF_PARAM


def _is_member_tuple(a: object) -> bool:
    return isinstance(a, tuple) and (
        len(a) == 0 or isinstance(a[0], (syntax.VariantDef, syntax.VariantRef))
    )


def _find_member_tuple(args: _Args) -> tuple[syntax.VariantDef | syntax.VariantRef, ...]:
    return cast(
        tuple[syntax.VariantDef | syntax.VariantRef, ...],
        next(a for a in args if _is_member_tuple(a)),
    )


def _is_case_branch_tuple(a: object) -> bool:
    return isinstance(a, tuple) and (len(a) == 0 or isinstance(a[0], syntax.CaseBranch))


def _find_case_branch_tuple(args: _Args) -> tuple[syntax.CaseBranch, ...]:
    return cast(tuple[syntax.CaseBranch, ...], next(a for a in args if _is_case_branch_tuple(a)))


def _require_literal_string(
    node: syntax.StringLit | syntax.Template, message: str
) -> syntax.StringLit:
    """Return *node* as a ``StringLit``, rejecting an interpolated ``Template``."""
    if isinstance(node, syntax.StringLit):
        return node
    raise AglSyntaxError(message, span=node.span)


def _is_expr_obj(a: object) -> bool:
    """Return True if *a* is an Expr (AST node, not a Token or None)."""
    return a is not None and not isinstance(a, Token)


def _is_expr_node(a: object) -> bool:
    """Return True if *a* is an expression AST node."""
    return isinstance(a, syntax.Expr) or isinstance(a, _RawInfixChain)


def _reject_positional_after_keyed(pos_arg_span: SourceSpan, *, seen_keyed: bool) -> None:
    """Raise if a positional argument follows a named/keyed one.

    Shared by arg_list and attr_arg_list.
    """
    if seen_keyed:
        raise AglSyntaxError(
            "positional argument after named argument is not allowed.",
            span=pos_arg_span,
        )


def _find_non_token(args: _Args) -> object:
    """Return the first non-None, non-Token element in args."""
    return next(a for a in args if a is not None and not isinstance(a, Token))


def _find_expr(args: _Args) -> syntax.Expr:
    """Return the first Expr in *args* (skip Tokens and None placeholders).

    Also used to extract single-Expr branch/suite bodies, which are likewise
    the sole non-token element in *args*.
    """
    return cast(syntax.Expr, _find_non_token(args))


def _extract_ann_and_optional_expr(
    tail: _Args,
) -> tuple[TypeExpr | None, syntax.Expr | None]:
    """Extract (type_ann, optional_expr) from a declaration or parameter tail.

    The tail is ``type_ann? EQ expr``, either half of which the grammar may
    leave out; ``maybe_placeholders=True`` spells a missing annotation ``None``.
    """
    ann: TypeExpr | None = None
    value: syntax.Expr | None = None
    for a in tail:
        if isinstance(a, _ALL_TYPE_EXPRS):
            ann = a
        elif _is_expr_obj(a):
            value = cast(syntax.Expr, a)
    return ann, value


def _extract_ann_and_value(
    tail: _Args,
) -> tuple[TypeExpr | None, syntax.Expr]:
    """Extract (type_ann, value) from a ``type_ann? EQ value`` tail."""
    ann = next((a for a in tail if isinstance(a, _ALL_TYPE_EXPRS)), None)
    return ann, cast(syntax.Expr, tail[-1])


def syntax_error_from_meta(meta: Meta, message: str) -> AglSyntaxError:
    """Create an AglSyntaxError from a Meta object."""
    return AglSyntaxError(message, span=_span_from_meta(meta))
