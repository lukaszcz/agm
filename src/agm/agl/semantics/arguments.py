"""Pure zone-binding algorithm shared by every AgL argument-binding call site.

This module holds the positional/named binding algorithm with no dependency on
the AST, source spans, or the typechecker: it maps a kind-annotated parameter
list against a call's positional and named arguments, generic over an opaque
item type ``T``.  ``agm.agl.typecheck.arguments.bind_arguments`` is the
AST-facing wrapper — it builds :class:`BindParam` lists from real parameter
signatures, delegates to :func:`bind_arguments` here, and translates
:class:`ArgumentBindingError` into ``AglTypeError`` with source spans and
messages.  That wrapper is what call expressions, constructors, and
constructor patterns use; this module exists so callers outside the
typechecker can share the same binding rule.

``kind`` is a :class:`ParamZone` value (``POSITIONAL_ONLY``, ``STANDARD``,
``NAMED_ONLY``) rather than the AST's ``ParamKind`` enum: ``ParamZone`` is
the IR-level zone enum from ``agm.agl.ir.zones``, re-exported here so
existing call sites keep importing it from this module; the typecheck
wrapper maps ``ParamKind`` to ``ParamZone`` at its boundary.

Algorithm (positional-greedy with named-only shorthand)
---------------------------------------------------------
1. **Positional args** (left to right):
   - While a positional-capable param (POSITIONAL_ONLY or STANDARD) remains
     unfilled, bind this arg to it — regardless of whether the arg is a bare
     name.
   - Once positional-capable params are exhausted, a further positional arg
     lands in **named-only territory**: it MUST be a bare name (per
     ``bare_name``); reinterpret it as the shorthand ``name = name``, binding
     the NAMED_ONLY param with that name.  A non-bare item there is an error.
     If no NAMED_ONLY param has that name, or it is already filled → error.
2. **Named args** (``name = value``): bind each to the param of that name.
   - Targeting a POSITIONAL_ONLY param by name → error.
   - Unknown name → error; already-filled (by a positional or a shorthand) →
     duplicate error.
3. **Defaults / missing**: any unfilled param with a default → ``None`` (use
   default); any unfilled param without a default → missing-required error.

The output erases the positional/named/shorthand distinction — each param maps
to a bound item or ``None`` (use-default marker), in declaration order.

Note that the named-only shorthand rule (reinterpreting a bare positional as
``name = name``) is itself part of this pure algorithm, because it governs how
positional and named-only arguments interact; what stays out of this module is
AST-call-site-only concerns like source spans and message text.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from agm.agl.ir.zones import ParamZone

T = TypeVar("T")

__all__ = [
    "ArgumentBindingError",
    "ArgumentBindingErrorKind",
    "BindParam",
    "ParamZone",
    "bind_arguments",
]

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BindParam:
    """Per-parameter descriptor for :func:`bind_arguments` (type-erased).

    The binder is type-agnostic: it only needs the name, zone, and whether the
    parameter has a default.  Type resolution is the caller's concern.
    """

    name: str
    kind: ParamZone
    has_default: bool


class ArgumentBindingErrorKind(enum.Enum):
    """Which zone-binding rule :func:`bind_arguments` found violated."""

    TOO_MANY_POSITIONAL = "too_many_positional"
    POSITIONAL_IN_NAMED_ONLY = "positional_in_named_only"
    POSITIONAL_ONLY_BY_NAME = "positional_only_by_name"
    UNKNOWN_NAME = "unknown_name"
    DUPLICATE = "duplicate"
    MISSING_REQUIRED = "missing_required"


class ArgumentBindingError(Exception):
    """Raised by :func:`bind_arguments` on any zone-binding violation.

    ``name`` is the offending or missing parameter/argument name, when the
    error concerns a name (every kind except the two positional-territory
    overflow kinds). ``positional_index`` is the offending item's position in
    *positional*, set when the error is sourced from a positional argument
    (``TOO_MANY_POSITIONAL``, ``POSITIONAL_IN_NAMED_ONLY``, and the
    named-only-shorthand cases of ``UNKNOWN_NAME``/``DUPLICATE``).
    ``named_index`` is the offending item's position in *named*, set when the
    error is sourced from a named argument (``POSITIONAL_ONLY_BY_NAME`` and
    the named-argument cases of ``UNKNOWN_NAME``/``DUPLICATE``). Both are
    ``None`` for ``MISSING_REQUIRED``, which has no offending argument item
    and locates on the call instead.
    """

    def __init__(
        self,
        kind: ArgumentBindingErrorKind,
        *,
        name: str | None = None,
        positional_index: int | None = None,
        named_index: int | None = None,
    ) -> None:
        super().__init__(kind.value)
        self.kind = kind
        self.name = name
        self.positional_index = positional_index
        self.named_index = named_index


# ---------------------------------------------------------------------------
# Core binding routine
# ---------------------------------------------------------------------------


def bind_arguments(
    params: Sequence[BindParam],
    positional: Sequence[T],
    named: Iterable[tuple[str, T]],
    *,
    bare_name: Callable[[T], str | None] = lambda _item: None,
) -> tuple[T | None, ...]:
    """Bind positional and named arguments against a kind-annotated parameter list.

    Parameters
    ----------
    params:
        The parameter list in declaration order (name, zone, has_default).
    positional:
        Positional argument items in source order.
    named:
        ``(name, item)`` pairs in source order (duplicate names are assumed to
        have been caught earlier by the caller), iterated exactly once — a
        caller may pass a ``dict.items()`` view.
    bare_name:
        A callable that returns the bare identifier string when the item is a
        bare-name reference, or ``None`` for any other item.  Used to
        implement the named-only shorthand rule.  Defaults to treating no
        item as bare-name-capable, for callers with no such notion.

    Returns
    -------
    A tuple in declaration order, one entry per parameter:
    - The bound argument item (``T``) if the argument was supplied.
    - ``None`` if the parameter's default should be used.

    Raises
    ------
    ArgumentBindingError
        On any binding violation: too many positional args, a non-bare
        positional arg in named-only territory, a positional-only param
        passed by name, an unknown name, a duplicate supply, or a missing
        required param.
    """
    bound: list[T | None] = [None] * len(params)
    filled: list[bool] = [False] * len(params)

    # --- Step 1: Bind positional args ---
    # Positional args fill positional-capable (POSITIONAL_ONLY/STANDARD) params in
    # declaration order.  A user-written zone-ordered list keeps these as a prefix,
    # but a constructor field list can interleave an inherited named-only field
    # (e.g. an exception's ``message``) before a marked positional-capable field, so
    # we advance past named-only params rather than assume a contiguous prefix.
    pos_idx = 0  # index into `params` for the next available positional-capable slot
    for i, arg in enumerate(positional):
        while pos_idx < len(params) and params[pos_idx].kind == ParamZone.NAMED_ONLY:
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
                has_named_only = any(p.kind == ParamZone.NAMED_ONLY for p in params)
                if not has_named_only:
                    raise ArgumentBindingError(
                        ArgumentBindingErrorKind.TOO_MANY_POSITIONAL, positional_index=i
                    )
                raise ArgumentBindingError(
                    ArgumentBindingErrorKind.POSITIONAL_IN_NAMED_ONLY, positional_index=i
                )
            # Find the named-only param with this name.
            target_idx: int | None = None
            for j, p in enumerate(params):
                if p.kind == ParamZone.NAMED_ONLY and p.name == name:
                    target_idx = j
                    break
            if target_idx is None:
                raise ArgumentBindingError(
                    ArgumentBindingErrorKind.UNKNOWN_NAME, name=name, positional_index=i
                )
            if filled[target_idx]:
                raise ArgumentBindingError(
                    ArgumentBindingErrorKind.DUPLICATE, name=name, positional_index=i
                )
            bound[target_idx] = arg
            filled[target_idx] = True

    # --- Step 2: Bind named args ---
    for named_index, (name, value) in enumerate(named):
        # Find the param with this name.
        target_idx = None
        for j, p in enumerate(params):
            if p.name == name:
                target_idx = j
                break
        if target_idx is None:
            raise ArgumentBindingError(
                ArgumentBindingErrorKind.UNKNOWN_NAME, name=name, named_index=named_index
            )
        p = params[target_idx]
        if p.kind == ParamZone.POSITIONAL_ONLY:
            raise ArgumentBindingError(
                ArgumentBindingErrorKind.POSITIONAL_ONLY_BY_NAME, name=name, named_index=named_index
            )
        if filled[target_idx]:
            raise ArgumentBindingError(
                ArgumentBindingErrorKind.DUPLICATE, name=name, named_index=named_index
            )
        bound[target_idx] = value
        filled[target_idx] = True

    # --- Step 3: Check defaults / missing ---
    for i, p in enumerate(params):
        if not filled[i] and not p.has_default:
            raise ArgumentBindingError(ArgumentBindingErrorKind.MISSING_REQUIRED, name=p.name)
        # else: bound[i] stays None → caller uses default.

    return tuple(bound)
