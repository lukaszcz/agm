"""Shared argument-binding routine for AgL function and constructor calls.

``bind_arguments`` is the ONE shared core that maps a kind-annotated parameter
list against a call's positional + named arguments.  It is generic over the
argument-item type ``T`` so the same function can serve:

- Call expressions (``T = Expr``) — used by the checker and the lowerer.
- Constructor patterns (``T = Pattern``) — used by the pattern checker (K4c).

The routine is PURE: it does not resolve types, does not lower expressions, and
does not mutate the checker state.  It raises ``AglTypeError`` on any binding
violation.

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

from agm.agl.syntax.nodes import ParamKind
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import AglTypeError

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
    # Split params into positional-capable prefix and named-only suffix.
    # Because zones are ordered (POSITIONAL_ONLY → STANDARD → NAMED_ONLY),
    # positional-capable params always form a contiguous prefix.
    n_pos_capable = sum(
        1
        for p in params
        if p.kind in (ParamKind.POSITIONAL_ONLY, ParamKind.STANDARD)
    )

    # Track which params have been bound (by param index).
    bound: list[T | None] = [None] * len(params)
    filled: list[bool] = [False] * len(params)

    # --- Step 1: Bind positional args ---
    pos_idx = 0  # index into `params` for the next available positional-capable slot
    for arg in positional:
        if pos_idx < n_pos_capable:
            # Bind to the next positional-capable parameter.
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
