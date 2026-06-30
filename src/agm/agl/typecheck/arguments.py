"""Shared argument-binding routines for AgL function and constructor calls.

``bind_arguments`` is the ONE shared core that maps a kind-annotated parameter
list against a call's positional + named arguments.  It is generic over the
argument-item type ``T`` so the same function can serve:

- Call expressions (``T = Expr``) — used by the checker and the lowerer.
- Constructor patterns (``T = Pattern``) — used by the pattern checker (K4c).

The routine is PURE: it does not resolve types, does not lower expressions, and
does not mutate the checker state.  It raises ``AglTypeError`` on any binding
violation.

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

Algorithm (positional-greedy with named-only shorthand)
-------------------------------------------------------
1. **Positional args** (left to right):
   - While a positional-capable param (POSITIONAL_ONLY or STANDARD) remains
     unfilled, bind this arg to it — regardless of whether the arg is a bare name.
   - Once positional-capable params are exhausted, a further positional arg lands
     in **named-only territory**: it MUST be a bare name (e.g. a ``VarRef``);
     reinterpret it as the shorthand ``name = name``, binding the NAMED_ONLY param
     with that name.  If the arg is not a bare name → ``AglTypeError``.  If no
     NAMED_ONLY param has that name, or it is already filled → error.
2. **Named args** (``name = value``): bind each to the param of that name.
   - Targeting a POSITIONAL_ONLY param by name → ``AglTypeError``.
   - Unknown name → error; already-filled (by a positional or a shorthand) → duplicate error.
3. **Defaults / missing**: any unfilled param with a default → ``None`` (use
   default); any unfilled param without a default → missing-required error.

The output erases the positional/named/shorthand distinction — each param maps
to a bound item or ``None`` (use-default marker), in declaration order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from agm.agl.syntax.nodes import (
    Expr,
    NamedArg,
    ParamKind,
    Pattern,
    PatternField,
    VarPattern,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import AglTypeError, ParamSpec

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
    kind: ParamKind
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
    # Track which params have been bound (by param index).
    bound: list[T | None] = [None] * len(params)
    filled: list[bool] = [False] * len(params)

    # --- Step 1: Bind positional args ---
    # Positional args fill positional-capable (POSITIONAL_ONLY/STANDARD) params in
    # declaration order.  A user-written zone-ordered list keeps these as a prefix,
    # but a constructor field list can interleave an inherited named-only field
    # (e.g. an exception's ``message``) before a marked positional-capable field, so
    # we advance past named-only params rather than assume a contiguous prefix.
    pos_idx = 0  # index into `params` for the next available positional-capable slot
    for arg in positional:
        while pos_idx < len(params) and params[pos_idx].kind == ParamKind.NAMED_ONLY:
            pos_idx += 1
        if pos_idx < len(params):
            # params[pos_idx] is positional-capable (POSITIONAL_ONLY or STANDARD).
            bound[pos_idx] = arg
            filled[pos_idx] = True
            pos_idx += 1
        else:
            # No positional-capable slots remain → named-only territory.
            # The arg MUST be a bare name (shorthand rule).
            name = bare_name(arg)
            if name is None:
                has_named_only = any(p.kind == ParamKind.NAMED_ONLY for p in params)
                if not has_named_only:
                    raise AglTypeError(
                        f"Too many positional arguments in {context_desc}.",
                        span=span_of(arg),
                    )
                raise AglTypeError(
                    f"Positional argument in a named-only position in {context_desc}. "
                    "Only a bare parameter name (shorthand 'name' for 'name = name') "
                    "is allowed here.",
                    span=span_of(arg),
                )
            # Find the named-only param with this name.
            target_idx: int | None = None
            for i, p in enumerate(params):
                if p.kind == ParamKind.NAMED_ONLY and p.name == name:
                    target_idx = i
                    break
            if target_idx is None:
                raise AglTypeError(
                    f"Unknown argument '{name}' in {context_desc}.",
                    span=span_of(arg),
                )
            if filled[target_idx]:
                raise AglTypeError(
                    f"Duplicate argument '{name}' in {context_desc}.",
                    span=span_of(arg),
                )
            bound[target_idx] = arg
            filled[target_idx] = True

    # --- Step 2: Bind named args ---
    for bn in named:
        # Find the param with this name.
        target_idx = None
        for i, p in enumerate(params):
            if p.name == bn.name:
                target_idx = i
                break
        if target_idx is None:
            raise AglTypeError(
                f"Unknown argument '{bn.name}' in {context_desc}.",
                span=bn.span,
            )
        p = params[target_idx]
        if p.kind == ParamKind.POSITIONAL_ONLY:
            raise AglTypeError(
                f"Parameter '{bn.name}' is positional-only and cannot be passed by name "
                f"in {context_desc}.",
                span=bn.span,
            )
        if filled[target_idx]:
            raise AglTypeError(
                f"Duplicate argument '{bn.name}' in {context_desc}.",
                span=bn.span,
            )
        bound[target_idx] = bn.value
        filled[target_idx] = True

    # --- Step 3: Check defaults / missing ---
    for i, p in enumerate(params):
        if not filled[i]:
            if not p.has_default:
                raise AglTypeError(
                    f"Missing required argument '{p.name}' in {context_desc}.",
                    span=call_span,
                )
            # bound[i] stays None → caller uses default.

    return tuple(bound)


# ---------------------------------------------------------------------------
# Constructor-specific convenience wrapper
# ---------------------------------------------------------------------------


def bind_constructor_args(
    field_kinds: tuple[tuple[str, ParamKind], ...],
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
        Ordered ``(field_name, ParamKind)`` pairs from the constructor's
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
    bound_exprs: dict[str, Expr] = {}
    for (fname, _fkind), bound_expr in zip(field_kinds, binding):
        assert bound_expr is not None, (
            f"compiler bug: required field '{fname}' missing in binding for {context_desc}"
        )
        bound_exprs[fname] = bound_expr
    return bound_exprs


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
    field_kinds: tuple[tuple[str, ParamKind], ...],
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
