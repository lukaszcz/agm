"""Compile-time folding of a module's own constants, from its AST alone.

A constant expression may name another constant of the same module and may
interpolate one into a template, so a declaration attribute can quote a value
the module already declares::

    let default-aspects = "correctness, maintainability"

    @doc("Defaults: %{default-aspects}")
    program def main() -> unit = ...

This module folds such an expression to its text. It reads one
:class:`~agm.agl.syntax.nodes.Program` and nothing else — no scope resolution,
no dependency graph, no type information — because the callers that need a
folded attribute include package command discovery
(:mod:`agm.packages.source_commands`), which scans a package's sources without
compiling them. That is also why a reference reaches the declaring module only:
a cross-module target would need the graph this leaf refuses to load.

Name resolution here mirrors the language's own rule for a same-module name:
a bare or scope-qualified reference is looked up in the
enclosing scope region, then outward level by level to the module root, while a
``::``-anchored one starts at the root. A module route is rejected outright.
Names an ``import`` or ``use`` contributes are not consulted — a module's own
declaration wins over them in ordinary resolution too, so the two agree
wherever this leaf answers at all. A local binding inside a function body never
participates: a reference in a constant position names a static binding.

Only scalars fold. Rendering an array, dict, or constructor needs the value
descriptors of a linked program, which this leaf has no access to, so naming
one in a hole is a fold failure rather than a second renderer.

``fold_scalar`` is the shared mechanics behind that folding — literal shape,
negation, ``not``, and template substitution — factored out so a caller with
its own reference resolution (a checked module's constant bindings, not just
this AST-only one) can fold a scalar the same way. Both resolvers gate a
reference on its binding's declared type before folding through it: absent,
or a scalar type (``text``/``int``/``decimal``/``bool``) directly, since any
other annotation (``json``, an alias, a generic instantiation, ...) may
render the binding's value differently from its initializer literal's own
text (coercion, wrapping) — folding through it would silently mis-render.

``constant_key`` builds on the same mechanics: the canonical, hashable
comparison key for a dict-literal key expression known to be constant, used
to detect a static duplicate key. Two keys compare equal exactly when AgL
equality would consider them equal — ``null`` is a singleton key, bool never
equals a number, int and decimal compare numerically, text compares after
folding (which includes ``not`` and negation over a folded operand), a
constant reference follows structurally to its initializer (a constructor
reference chases through a chain of constant aliases too), and a call
compares by its identity (a constructor or a constant builtin) with its
arguments normalized to field names.
"""

from __future__ import annotations

import decimal
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

from agm.agl.syntax.nodes import (
    ArrayLit,
    BoolLit,
    BuiltinVarDecl,
    Call,
    CompleteCall,
    DecimalLit,
    DictLit,
    EnumDef,
    ExceptionDef,
    Expr,
    FuncDef,
    InterpSegment,
    IntLit,
    LetDecl,
    NullLit,
    Program,
    QualifierAnchor,
    RecordDef,
    StringLit,
    Template,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarRef,
    is_complete_call,
    static_items,
)
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    BoolT,
    DecimalT,
    IntT,
    TextT,
    TypeExpr,
    render_qualified_name,
    render_type_expr,
)
from agm.agl.value_syntax.lexical import scalar_text

__all__ = ["FoldFailure", "ModuleConstants", "Scalar", "constant_key", "fold_scalar"]

#: What a hole may fold to: the value kinds AgL spells as plain text.
Scalar = str | int | decimal.Decimal | bool

#: The type-expression shapes a binding's annotation may carry and still
#: permit folding through it — see module docstring.
_SCALAR_TYPE_EXPRS = (TextT, IntT, DecimalT, BoolT)


def _scalar_fold_annotation_ok(annotation: TypeExpr | None) -> bool:
    """Whether a binding's declared type still permits folding its initializer as a scalar."""
    return annotation is None or isinstance(annotation, _SCALAR_TYPE_EXPRS)


#: A binding's declaration path: its scope path, then its name.
_BindingKey = tuple[tuple[str, ...], str]

#: Static items that carry the scope path they were declared at. Every other
#: item — a bare root expression, an ``import`` — declares no scoped name.
_SCOPED_ITEMS = (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    FuncDef,
    LetDecl,
    RecordDef,
    TypeAlias,
    VarDecl,
)


@dataclass(frozen=True, slots=True)
class FoldFailure:
    """Why an expression could not be folded to a constant.

    ``message`` is a sentence fragment naming the obstruction, for a caller to
    phrase into its own diagnostic; ``span`` locates the offending
    sub-expression rather than the whole expression.
    """

    message: str
    span: SourceSpan


def fold_scalar(
    expr: Expr, *, resolve_ref: Callable[[VarRef], Scalar | FoldFailure]
) -> Scalar | FoldFailure:
    """Fold *expr* to the scalar it denotes, or say why it does not fold.

    Handles the mechanics common to every caller: literal shape, negation,
    ``not``, and template substitution (a non-text hole renders through
    ``scalar_text``, exactly like interpolation). ``resolve_ref`` is the
    caller's own reference resolution — an AST-only same-module scope walk
    (:class:`ModuleConstants`) or a checked module's constant bindings — which
    may chase a chain of further references, with its own cycle detection,
    before returning the scalar a ``VarRef`` ultimately denotes, or a failure
    explaining why it names none.
    """
    if isinstance(expr, StringLit):
        return expr.value
    if isinstance(expr, (IntLit, DecimalLit, BoolLit)):
        return expr.value
    if isinstance(expr, UnaryNeg):
        operand = fold_scalar(expr.operand, resolve_ref=resolve_ref)
        if isinstance(operand, FoldFailure):
            return operand
        if isinstance(operand, bool) or not isinstance(operand, (int, decimal.Decimal)):
            return FoldFailure("'-' applies to a number only", expr.span)
        return -operand
    if isinstance(expr, UnaryNot):
        operand = fold_scalar(expr.operand, resolve_ref=resolve_ref)
        if isinstance(operand, FoldFailure):
            return operand
        if not isinstance(operand, bool):
            return FoldFailure("'not' applies to a bool only", expr.span)
        return not operand
    if isinstance(expr, Template):
        parts: list[str] = []
        for segment in expr.segments:
            if not isinstance(segment, InterpSegment):
                parts.append(segment.text)
                continue
            hole = fold_scalar(segment.expr, resolve_ref=resolve_ref)
            if isinstance(hole, FoldFailure):
                return hole
            parts.append(hole if isinstance(hole, str) else scalar_text(hole))
        return "".join(parts)
    if isinstance(expr, VarRef):
        return resolve_ref(expr)
    if isinstance(expr, (ArrayLit, DictLit, NullLit, UnitLit)):
        return FoldFailure("only a text, int, decimal, or bool constant folds into text", expr.span)
    return FoldFailure("it is not a constant expression", expr.span)


@dataclass(frozen=True, slots=True)
class _BoolKey:
    value: bool


@dataclass(frozen=True, slots=True)
class _NumKey:
    value: decimal.Decimal


@dataclass(frozen=True, slots=True)
class _NullKey:
    pass


@dataclass(frozen=True, slots=True)
class _TextKey:
    value: str


@dataclass(frozen=True, slots=True)
class _CallKey:
    """A constructor application or a constant builtin call: identity plus bound fields."""

    identity: Hashable
    fields: tuple[tuple[str, Hashable], ...]


def _field_name(item: tuple[str, Hashable]) -> str:
    """Sort key for a ``(field_name, key)`` pair, typed to avoid an ``Any`` lambda."""
    return item[0]


def constant_key(
    expr: Expr,
    *,
    initializer_for: Callable[[int], tuple[Expr, TypeExpr | None] | None],
    constructor_ref_for: Callable[[VarRef], Hashable | None],
    call_binding: Callable[[CompleteCall], tuple[Hashable, Mapping[str, Expr]] | None],
) -> Hashable | None:
    """Canonical comparison key for a constant dict-literal key expression.

    Two constant keys compare equal under this key exactly when AgL equality
    would consider them equal: ``null`` is a singleton key; a bool never
    equals a number; int and decimal literals compare numerically; text
    compares after folding templates and constant references (through
    negation and ``not``, via ``fold_scalar``); a reference to a constant of
    the module follows to its initializer structurally; a call — a
    constructor application, direct or through a chain of constant aliases,
    or a constant builtin call — compares by its identity with its arguments
    normalized to field names (positional and named alike, via
    ``call_binding``).

    Returns ``None`` when *expr* has no comparable structural form (an
    interpolation hole reading something other than a scalar-annotated
    constant, an ordinary function call, ...), leaving such a key to a
    runtime ``DuplicateKeyError`` only — this never produces a false
    positive.

    Must be called only on an expression already known constant
    (``is_constant_expression``): a module-constant reference it reaches is
    then guaranteed acyclic, since classifying the whole expression constant
    already walked — and would have rejected — any cycle.

    ``initializer_for`` resolves a constant ``VarRef``'s ``node_id`` to its
    initializer expression and declared type, or ``None`` when it does not
    name a fixed (non-``var``, non-``@param``) binding.
    ``constructor_ref_for`` resolves a bare constructor-denoting ``VarRef`` to
    its identity, or ``None`` when it does not denote a constructor.
    ``call_binding`` resolves a call whose callee is a ``VarRef`` — a
    constructor call (direct or through an alias) or a constant builtin call
    — to its identity and its arguments already bound to field names, or
    ``None`` when the callee denotes neither.
    """

    def recur(sub_expr: Expr) -> Hashable | None:
        return constant_key(
            sub_expr,
            initializer_for=initializer_for,
            constructor_ref_for=constructor_ref_for,
            call_binding=call_binding,
        )

    def resolve_ref(ref: VarRef) -> Scalar | FoldFailure:
        resolved = initializer_for(ref.node_id)
        if resolved is None:
            return FoldFailure(f"{_spelling(ref)!r} names no constant", ref.span)
        target, annotation = resolved
        if annotation is not None and not _scalar_fold_annotation_ok(annotation):
            return FoldFailure(
                f"{_spelling(ref)!r} is annotated {render_type_expr(annotation)!r}, not a "
                "text, int, decimal, or bool constant",
                ref.span,
            )
        return fold_scalar(target, resolve_ref=resolve_ref)

    if isinstance(expr, NullLit):
        return _NullKey()
    if isinstance(expr, VarRef):
        ctor_id = constructor_ref_for(expr)
        if ctor_id is not None:
            return _CallKey(ctor_id, ())
        resolved = initializer_for(expr.node_id)
        return None if resolved is None else recur(resolved[0])
    if isinstance(expr, Call) and isinstance(expr.callee, VarRef) and is_complete_call(expr):
        binding = call_binding(expr)
        if binding is None:
            return None
        identity, bound = binding
        return _call_key(identity, bound, recur)
    folded = fold_scalar(expr, resolve_ref=resolve_ref)
    if isinstance(folded, FoldFailure):
        return None
    if isinstance(folded, bool):
        return _BoolKey(folded)
    if isinstance(folded, str):
        return _TextKey(folded)
    return _NumKey(decimal.Decimal(folded))


def _call_key(
    identity: Hashable,
    bound: Mapping[str, Expr],
    recur: Callable[[Expr], Hashable | None],
) -> Hashable | None:
    """A call's key: its identity plus each bound field's key, sorted by name."""
    values: dict[str, Hashable] = {}
    for name, arg in bound.items():
        sig = recur(arg)
        if sig is None:
            return None
        values[name] = sig
    return _CallKey(identity, tuple(sorted(values.items(), key=_field_name)))


class ModuleConstants:
    """One module's static bindings, folded on demand.

    Built once per module AST and queried per constant position. Folding is
    memo-free: a constant position is rare and each fold is shallow, so a
    binding referenced twice is simply folded twice.
    """

    def __init__(self, program: Program) -> None:
        self._bindings: dict[_BindingKey, LetDecl | VarDecl] = {}
        self._item_scopes: dict[int, tuple[str, ...]] = {}
        for item in static_items(program.body.items):
            if not isinstance(item, _SCOPED_ITEMS):
                continue
            path = tuple(segment.name for segment in item.scope_path)
            self._item_scopes[item.node_id] = path
            if isinstance(item, (LetDecl, VarDecl)):
                self._bindings[(path, item.name)] = item

    def static_scope_path_of(self, node_id: int) -> tuple[str, ...] | None:
        """Return the scope path a static item's own references resolve from.

        ``None`` for a node that is not one of this module's static items — a
        parameter, a field, or a declaration nested in a function body — whose
        references resolve from the scope path of the static item enclosing it.
        """
        return self._item_scopes.get(node_id)

    def fold_text(self, expr: Expr, *, scope_path: tuple[str, ...]) -> str | FoldFailure:
        """Fold *expr*, referenced from *scope_path*, to its constant text.

        *expr* itself must be of text type — a literal, a template, or a
        reference to one — the way any other expression checked against a
        declared type would be. A number or bool folds only inside a hole,
        where interpolation renders it.
        """
        folded = fold_scalar(expr, resolve_ref=self._resolver(scope_path, ()))
        if isinstance(folded, FoldFailure):
            return folded
        if not isinstance(folded, str):
            return FoldFailure("it is not a text constant", expr.span)
        return folded

    def _resolver(
        self, scope_path: tuple[str, ...], pending: tuple[_BindingKey, ...]
    ) -> Callable[[VarRef], Scalar | FoldFailure]:
        """Build the ``fold_scalar`` reference resolver rooted at *scope_path*.

        ``pending`` is the chain of bindings whose initializers are being
        folded, so a constant that names itself is reported as the cycle it is
        rather than recursing. Resolving a reference to another binding
        recurses through a fresh resolver rooted at that binding's own scope,
        since an unqualified name inside its initializer resolves from there.
        """

        def resolve(expr: VarRef) -> Scalar | FoldFailure:
            qualifier = expr.qualifier
            if qualifier is not None and (
                qualifier.anchored or any("/" in segment.name for segment in qualifier.segments)
            ):
                return FoldFailure(
                    f"{_spelling(expr)!r} names another module; a constant may name only "
                    "constants of its own module",
                    expr.span,
                )
            key = self._resolve(expr, scope_path)
            if key is None:
                return FoldFailure(
                    f"{_spelling(expr)!r} names no constant of this module", expr.span
                )
            if key in pending:
                return FoldFailure(f"{_spelling(expr)!r} is defined in terms of itself", expr.span)
            binding = self._bindings[key]
            annotation = binding.type_ann
            if annotation is not None and not _scalar_fold_annotation_ok(annotation):
                return FoldFailure(
                    f"{_spelling(expr)!r} is annotated {render_type_expr(annotation)!r}, "
                    "not a text, int, decimal, or bool constant",
                    expr.span,
                )
            return fold_scalar(binding.value, resolve_ref=self._resolver(key[0], (*pending, key)))

        return resolve

    def _resolve(self, expr: VarRef, scope_path: tuple[str, ...]) -> _BindingKey | None:
        """Resolve one reference to a static binding of this module.

        This is a deliberate AST-only approximation of the language's own
        resolution of a same-module name, not a second resolver: an attribute
        is folded before references are resolved, and package command discovery
        scans sources it never compiles. *expr* has already been screened for a
        module route, so its qualifier is a scope path: relative to
        *scope_path* and every enclosing level, or rooted when it is
        ``::``-anchored.
        """
        qualifier = expr.qualifier
        if qualifier is None:
            segments: tuple[str, ...] = ()
            rooted = False
        else:
            segments = tuple(segment.name for segment in qualifier.segments)
            rooted = qualifier.anchor is QualifierAnchor.CURRENT_MODULE
        for base in enclosing_scope_bases(scope_path, rooted=rooted):
            key = ((*base, *segments), expr.name)
            if key in self._bindings:
                return key
        return None


def _spelling(expr: VarRef) -> str:
    """Return a reference as it was written, for a diagnostic."""
    return render_qualified_name(expr.qualifier, expr.name)
