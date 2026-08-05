"""Render a semantic ``Type`` as the AgL type-annotation syntax a program would write.

This is a *partial* renderer, distinct from ``Type.__repr__`` (which is total and
exists purely for debugging): it covers every type form that has a surface
spelling and raises :class:`TypeSyntaxError` for the ones that do not — an
inference variable, ``bottom``, or a rigid type variable used outside the
generic declaration that binds it. Nothing here changes ``__repr__``; the two
serve different, deliberately separate purposes.
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.modules.ids import ModuleId
from agm.agl.semantics.types import (
    ArrayType,
    BoolType,
    BottomType,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
    spells_bare,
)

__all__ = ["TypeSyntaxError", "render_type_syntax"]


class TypeSyntaxError(ValueError):
    """*type*, or a type nested within it, has no AgL surface spelling."""


def render_type_syntax(t: Type) -> str:
    """Render *t* as the type-annotation text an AgL program would write for it.

    Covers every type form with a surface spelling: the built-in scalars,
    ``array``/``dict`` containers, function types, and module- and
    scope-qualified nominal (record/enum/exception) types, including generic
    instantiation. Qualification follows :func:`agm.agl.semantics.types
    .spells_bare` — the same rule ``RecordType``/``EnumType``/
    ``ExceptionType.__repr__`` use for diagnostics: a program's own
    declarations and the standard library's own built-ins spell bare, every
    other nominal type needs its module path.

    Raises:
        TypeSyntaxError: if *t*, or a type nested within it, is an inference
            variable, ``bottom``, or a rigid type variable — none of these has
            a way to be written in a program outside the generic declaration
            that binds it.
    """
    if isinstance(t, TextType):
        return "text"
    if isinstance(t, JsonType):
        return "json"
    if isinstance(t, BoolType):
        return "bool"
    if isinstance(t, IntType):
        return "int"
    if isinstance(t, DecimalType):
        return "decimal"
    if isinstance(t, UnitType):
        return "unit"
    if isinstance(t, ArrayType):
        return f"array[{render_type_syntax(t.elem)}]"
    if isinstance(t, DictType):
        return f"dict[text, {render_type_syntax(t.value)}]"
    if isinstance(t, (RecordType, EnumType)):
        return _render_nominal(t.module_id, t.scope_path, t.name, t.type_args)
    if isinstance(t, ExceptionType):
        return _render_nominal(t.module_id, t.scope_path, t.name, ())
    if isinstance(t, FunctionType):
        return _render_function(t)
    if isinstance(t, (BottomType, TypeVarType, InferenceVarType)):
        raise TypeSyntaxError(f"{t!r} has no AgL surface spelling.")
    assert_never(t)  # pragma: no cover


def _render_nominal(
    module_id: ModuleId, scope_path: tuple[str, ...], name: str, type_args: tuple[Type, ...]
) -> str:
    prefix = "" if spells_bare(module_id, name) else f"{module_id.path_str()}::"
    scoped_name = "::".join((*scope_path, name))
    if not type_args:
        return f"{prefix}{scoped_name}"
    args = ", ".join(render_type_syntax(arg) for arg in type_args)
    return f"{prefix}{scoped_name}[{args}]"


def _render_function(t: FunctionType) -> str:
    result = render_type_syntax(t.result)
    if not t.params:
        return f"() -> {result}"
    if len(t.params) == 1 and not isinstance(t.params[0], FunctionType):
        return f"{render_type_syntax(t.params[0])} -> {result}"
    params = ", ".join(render_type_syntax(param) for param in t.params)
    return f"({params}) -> {result}"
