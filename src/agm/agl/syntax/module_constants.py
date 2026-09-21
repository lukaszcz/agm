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
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass

from agm.agl.syntax.nodes import (
    ArrayLit,
    BoolLit,
    BuiltinVarDecl,
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
    static_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.value_syntax.lexical import scalar_text

__all__ = ["FoldFailure", "ModuleConstants", "Scalar"]

#: What a hole may fold to: the value kinds AgL spells as plain text.
Scalar = str | int | decimal.Decimal | bool

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
        folded = self._fold(expr, scope_path, ())
        if isinstance(folded, FoldFailure):
            return folded
        if not isinstance(folded, str):
            return FoldFailure("it is not a text constant", expr.span)
        return folded

    def _fold(
        self, expr: Expr, scope_path: tuple[str, ...], pending: tuple[_BindingKey, ...]
    ) -> Scalar | FoldFailure:
        """Fold one expression to a scalar, or say why it does not fold.

        ``pending`` is the chain of bindings whose initializers are being
        folded, so a constant that names itself is reported as the cycle it is
        rather than recursing.
        """
        if isinstance(expr, StringLit):
            return expr.value
        if isinstance(expr, (IntLit, DecimalLit, BoolLit)):
            return expr.value
        if isinstance(expr, UnaryNeg):
            return self._negate(expr, scope_path, pending)
        if isinstance(expr, UnaryNot):
            operand = self._fold(expr.operand, scope_path, pending)
            if isinstance(operand, FoldFailure):
                return operand
            if not isinstance(operand, bool):
                return FoldFailure("'not' applies to a bool only", expr.span)
            return not operand
        if isinstance(expr, Template):
            return self._fold_template(expr, scope_path, pending)
        if isinstance(expr, VarRef):
            return self._fold_reference(expr, scope_path, pending)
        if isinstance(expr, (ArrayLit, DictLit, NullLit, UnitLit)):
            return FoldFailure(
                "only a text, int, decimal, or bool constant folds into text", expr.span
            )
        return FoldFailure("it is not a constant expression", expr.span)

    def _negate(
        self, expr: UnaryNeg, scope_path: tuple[str, ...], pending: tuple[_BindingKey, ...]
    ) -> Scalar | FoldFailure:
        operand = self._fold(expr.operand, scope_path, pending)
        if isinstance(operand, FoldFailure):
            return operand
        if isinstance(operand, bool) or not isinstance(operand, (int, decimal.Decimal)):
            return FoldFailure("'-' applies to a number only", expr.span)
        return -operand

    def _fold_template(
        self, expr: Template, scope_path: tuple[str, ...], pending: tuple[_BindingKey, ...]
    ) -> str | FoldFailure:
        parts: list[str] = []
        for segment in expr.segments:
            if not isinstance(segment, InterpSegment):
                parts.append(segment.text)
                continue
            hole = self._fold(segment.expr, scope_path, pending)
            if isinstance(hole, FoldFailure):
                return hole
            parts.append(hole if isinstance(hole, str) else scalar_text(hole))
        return "".join(parts)

    def _fold_reference(
        self, expr: VarRef, scope_path: tuple[str, ...], pending: tuple[_BindingKey, ...]
    ) -> Scalar | FoldFailure:
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
            return FoldFailure(f"{_spelling(expr)!r} names no constant of this module", expr.span)
        if key in pending:
            return FoldFailure(f"{_spelling(expr)!r} is defined in terms of itself", expr.span)
        binding = self._bindings[key]
        return self._fold(binding.value, key[0], (*pending, key))

    def _resolve(self, expr: VarRef, scope_path: tuple[str, ...]) -> _BindingKey | None:
        """Resolve one reference to a static binding of this module.

        *expr* has already been screened for a module route, so its qualifier
        is a scope path: relative to *scope_path* and every enclosing level, or
        rooted when it is ``::``-anchored.
        """
        qualifier = expr.qualifier
        if qualifier is None:
            segments: tuple[str, ...] = ()
            anchored = False
        else:
            segments = tuple(segment.name for segment in qualifier.segments)
            anchored = qualifier.anchor is QualifierAnchor.CURRENT_MODULE
        levels: tuple[tuple[str, ...], ...] = (
            ((),)
            if anchored
            else tuple(scope_path[:depth] for depth in range(len(scope_path), -1, -1))
        )
        for base in levels:
            key = ((*base, *segments), expr.name)
            if key in self._bindings:
                return key
        return None


def _spelling(expr: VarRef) -> str:
    """Return a reference as it was written, for a diagnostic."""
    qualifier = expr.qualifier
    if qualifier is None:
        return expr.name
    prefix = "/" if qualifier.anchored else ""
    route = "/".join(qualifier.route_segments)
    if qualifier.anchored or "/" in route:
        return f"{prefix}{route}::{expr.name}"
    anchor = "::" if qualifier.anchor is QualifierAnchor.CURRENT_MODULE else ""
    return anchor + "::".join((*(segment.name for segment in qualifier.segments), expr.name))
