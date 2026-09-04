"""AST-facing argument-binding routines for AgL function and constructor calls.

``bind_arguments`` is the ONE shared entry point AST-facing callers use to map
a kind-annotated parameter list against a call's positional + named
arguments.  It is generic over the argument-item type ``T`` so the same
function can serve:

- Call expressions (``T = Expr``) — used by the checker and the lowerer.
- Constructor patterns (``T = Pattern``) — used by the pattern checker.

The zone-binding algorithm itself is pure (no AST, span, or checker
dependency) and lives in ``agm.agl.semantics.arguments``, so callers outside
the typechecker can share the same binding rule.  This module is the
AST-facing wrapper: it builds the pure module's ``BindParam`` list from real
parameter signatures, delegates to its ``bind_arguments``, and translates its
``ArgumentBindingError`` into ``AglTypeError`` with source spans and
messages.  It does not resolve types, does not lower expressions, and does
not mutate the checker state.

``bind_constructor_args`` and ``bind_call_args`` are convenience wrappers over
``bind_arguments`` for the two common cases.  ``bind_constructor_args`` builds
the ``BindParam`` list from a field-kinds tuple (record, enum variant,
exception), asserts every field is bound (constructors have no defaults), and
returns the ``{field_name: expr}`` mapping in declaration order.
``bind_call_args`` builds the ``BindParam`` list from a function's
``ParamSpec`` sequence and returns the declaration-order binding tuple (a bound
expression or ``None`` to use the parameter's default).  Both apply the VarRef
bare-name rule and are used by the checker (concrete and generic-inference
paths) and the lowerer to avoid repeating the same boilerplate at each site.

See ``agm.agl.semantics.arguments`` for the zone-binding algorithm itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, assert_never

from agm.agl.semantics import arguments as pure
from agm.agl.syntax.nodes import (
    Expr,
    NamedArg,
    Pattern,
    PatternField,
    VarPattern,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import AglTypeError, ParamSpec
from agm.agl.zones import ParamZone

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindParam:
    """Per-parameter descriptor for ``bind_arguments`` (type-erased).

    The binder is type-agnostic: it only needs the name, zone, and whether
    the parameter has a default.  Type resolution is the caller's concern.
    """

    name: str
    kind: ParamZone
    has_default: bool


@dataclass(frozen=True, slots=True)
class BoundName(Generic[T]):
    """A named argument ``name = value`` with the source span of the item.

    ``span`` is used to attach errors to the right source location.
    """

    name: str
    value: T
    span: SourceSpan


# ---------------------------------------------------------------------------
# Core binding routine
# ---------------------------------------------------------------------------


def bind_arguments(
    params: Sequence[BindParam],
    positional: Sequence[T],
    named: Sequence[BoundName[T]],
    *,
    bare_name: Callable[[T], str | None],
    span_of: Callable[[T], SourceSpan],
    call_span: SourceSpan,
    context_desc: str,
) -> tuple[T | None, ...]:
    """Bind positional and named arguments against a kind-annotated parameter list.

    Parameters
    ----------
    params:
        The parameter list in declaration order (name, kind, has_default).
    positional:
        Positional argument items in source order.
    named:
        Named argument items in source order (duplicate names are assumed to
        have been caught earlier by the parser/transformer).
    bare_name:
        A callable that returns the bare identifier string when the item is a
        bare-name reference (e.g. a ``VarRef``), or ``None`` for any other item.
        Used to implement the named-only shorthand rule.
    span_of:
        A callable that returns the source span of an argument item.  Used to
        attach errors to the right source location.
    call_span:
        The span of the entire call expression (used when no per-item span is
        available, e.g. for missing-arg errors).
    context_desc:
        Human-readable description of the call site for error messages, e.g.
        ``"call to 'f'"``.

    Returns
    -------
    A tuple in declaration order, one entry per parameter:
    - The bound argument item (``T``) if the argument was supplied.
    - ``None`` if the parameter's default should be used.

    Raises
    ------
    AglTypeError
        On any binding violation: too many positional args, non-bare positional
        arg in named-only territory, positional-only param passed by name,
        unknown named arg, duplicate supply (positional + named), or missing
        required param.
    """
    pure_params = [
        pure.BindParam(name=p.name, kind=p.kind, has_default=p.has_default) for p in params
    ]
    pure_named = [(bn.name, bn.value) for bn in named]
    try:
        return pure.bind_arguments(pure_params, positional, pure_named, bare_name=bare_name)
    except pure.ArgumentBindingError as err:
        raise _to_agl_type_error(
            err,
            params,
            positional,
            named,
            span_of=span_of,
            call_span=call_span,
            context_desc=context_desc,
        ) from err


def _to_agl_type_error(
    err: pure.ArgumentBindingError,
    params: Sequence[BindParam],
    positional: Sequence[T],
    named: Sequence[BoundName[T]],
    *,
    span_of: Callable[[T], SourceSpan],
    call_span: SourceSpan,
    context_desc: str,
) -> AglTypeError:
    """Translate a pure ``ArgumentBindingError`` into an ``AglTypeError``.

    Reproduces the exact messages and span placement of the inline checks the
    binding algorithm used to raise directly, before it moved to
    ``agm.agl.semantics.arguments``.
    """
    if err.positional_index is not None:
        span = span_of(positional[err.positional_index])
    elif err.named_index is not None:
        span = named[err.named_index].span
    else:
        span = call_span

    match err.kind:
        case pure.ArgumentBindingErrorKind.TOO_MANY_POSITIONAL:
            return AglTypeError(f"Too many positional arguments in {context_desc}.", span=span)
        case pure.ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY:
            return AglTypeError(
                f"Positional argument in a named-only position in {context_desc}. "
                "Only a bare parameter name (shorthand 'name' for 'name = name') "
                "is allowed here.",
                span=span,
            )
        case pure.ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME:
            return AglTypeError(
                f"Parameter '{err.name}' is positional-only and cannot be passed by name "
                f"in {context_desc}.",
                span=span,
            )
        case pure.ArgumentBindingErrorKind.UNKNOWN_NAME:
            return AglTypeError(f"Unknown argument '{err.name}' in {context_desc}.", span=span)
        case pure.ArgumentBindingErrorKind.DUPLICATE:
            return AglTypeError(f"Duplicate argument '{err.name}' in {context_desc}.", span=span)
        case pure.ArgumentBindingErrorKind.MISSING_REQUIRED:
            return AglTypeError(
                f"Missing required argument '{err.name}' in {context_desc}.", span=span
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Constructor-specific convenience wrapper
# ---------------------------------------------------------------------------


def bind_constructor_args(
    field_kinds: tuple[tuple[str, ParamZone], ...],
    positional: Sequence[Expr],
    named: Sequence[NamedArg],
    *,
    call_span: SourceSpan,
    context_desc: str,
) -> dict[str, Expr]:
    """Bind positional and named arguments for a record/enum/exception constructor.

    Builds the :class:`BindParam` list from *field_kinds*, runs
    :func:`bind_arguments` with the ``VarRef`` bare-name rule, asserts every
    field is bound (constructors have no defaults), and returns an ordered
    ``{field_name: expr}`` mapping.

    Parameters
    ----------
    field_kinds:
        Ordered ``(field_name, ParamZone)`` pairs from the constructor's
        field-kinds registry — produced by ``get_constructor_field_kinds``.
    positional:
        Positional argument expressions in source order.
    named:
        Named argument expressions in source order.
    call_span:
        The span of the entire call expression (for missing-arg errors).
    context_desc:
        Human-readable description for error messages, e.g.
        ``"constructor 'R'"``, ``"variant 'E.Some'"``, ``"exception 'Boom'"``.

    Returns
    -------
    An ordered ``{field_name: Expr}`` dict mapping each declared field to its
    bound argument expression.  The dict is in field declaration order.

    Raises
    ------
    AglTypeError
        On any binding violation (surplus/missing/duplicate args, positional
        arg in named-only territory that is not a bare name, etc.).
    """
    bind_params = tuple(
        BindParam(name=fname, kind=fkind, has_default=False) for fname, fkind in field_kinds
    )
    named_bns: list[BoundName[Expr]] = [
        BoundName(name=na.name, value=na.value, span=na.span) for na in named
    ]
    binding = bind_arguments(
        bind_params,
        positional,
        named_bns,
        bare_name=lambda e: e.name if isinstance(e, VarRef) else None,
        span_of=lambda e: e.span,
        call_span=call_span,
        context_desc=context_desc,
    )
    # Every field has has_default=False, so bind_arguments either fills every
    # entry or raises MISSING_REQUIRED — the None branch is unreachable here.
    return {
        fname: bound_expr
        for (fname, _fkind), bound_expr in zip(field_kinds, binding)
        if bound_expr is not None
    }


# ---------------------------------------------------------------------------
# Function-call convenience wrapper
# ---------------------------------------------------------------------------


def bind_call_args(
    params: Sequence[ParamSpec],
    positional: Sequence[Expr],
    named: Sequence[NamedArg],
    *,
    call_span: SourceSpan,
    context_desc: str,
) -> tuple[Expr | None, ...]:
    """Bind positional and named arguments for a function call against *params*.

    Builds the :class:`BindParam` list from a function's :class:`ParamSpec`
    sequence, applies the ``VarRef`` bare-name rule, and returns the
    declaration-order binding tuple from :func:`bind_arguments` (each entry is
    the bound expression, or ``None`` to use the parameter's default).

    Shared by the checker (concrete check and generic-inference paths) and the
    lowerer so the bare-name shorthand is always matched to its parameter by
    NAME rather than by raw positional index.  Raises ``AglTypeError`` on any
    binding violation.
    """
    bind_params = tuple(
        BindParam(name=p.name, kind=p.kind, has_default=p.has_default) for p in params
    )
    named_bns = [BoundName(name=na.name, value=na.value, span=na.span) for na in named]
    return bind_arguments(
        bind_params,
        positional,
        named_bns,
        bare_name=lambda e: e.name if isinstance(e, VarRef) else None,
        span_of=lambda e: e.span,
        call_span=call_span,
        context_desc=context_desc,
    )


# ---------------------------------------------------------------------------
# Constructor-pattern convenience wrapper
# ---------------------------------------------------------------------------


def bind_pattern_args(
    field_kinds: tuple[tuple[str, ParamZone], ...],
    positional: Sequence[Pattern],
    named: Sequence[PatternField],
    *,
    call_span: SourceSpan,
    context_desc: str,
) -> tuple[Pattern | None, ...]:
    """Bind positional and named sub-patterns for a constructor pattern.

    Builds the :class:`BindParam` list from *field_kinds*, applies the
    ``VarPattern`` bare-name rule, and returns the declaration-order binding
    tuple from :func:`bind_arguments`.  Unlike the call/constructor wrappers,
    every field is treated as defaulted (``has_default=True``) because
    constructor patterns may be partial — unmentioned fields stay ``None`` and
    the caller treats them as wildcards.
    """
    bind_params = tuple(
        BindParam(name=fname, kind=fkind, has_default=True) for fname, fkind in field_kinds
    )
    named_bns: list[BoundName[Pattern]] = [
        BoundName(name=pf.name, value=pf.pattern, span=pf.span) for pf in named
    ]
    return bind_arguments(
        bind_params,
        positional,
        named_bns,
        bare_name=lambda p: p.name if isinstance(p, VarPattern) else None,
        span_of=lambda p: p.span,
        call_span=call_span,
        context_desc=context_desc,
    )
