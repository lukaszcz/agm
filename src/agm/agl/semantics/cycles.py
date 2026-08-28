"""Cycle detection shared by the AgL structured-value walkers.

Reference semantics makes cyclic ``array``/``dict`` and record values
constructible: an array, dict, or mutable record field can hold a reference
back to a structured value that (transitively) contains it. Every walker that
recurses through structured values — rendering and JSON serialization — must
detect that re-entry rather than recursing forever. Arrays, dicts, records,
enum members, and exceptions therefore join the active set; scalar values
never need to join it.

The FFI encoder does not walk array or dict payloads: it produces lazy views,
so cyclic arguments cross the boundary. A companion that ``repr``s such a
view renders its value and reaches this guard. This module is the single
shared implementation for ``render_value`` and ``value_to_json_obj``.
Cyclic *Python* payloads are a separate concern with a separate walk in
``runtime/boundary.py``: they are unrepresentable rather than
cycle-guarded, and are rejected alongside the rest of the JSON-shape check.
Equality is unrelated — it is co-inductive (``semantics/values.py``) rather
than error-raising, and does not use this module.
"""

from __future__ import annotations

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception

__all__ = [
    "CYCLE_MESSAGE",
    "CYCLIC_VALUE_MARKER",
    "AglCyclicValue",
    "cyclic_value_raise",
    "enter_value",
]

#: The single ``CyclicValueError`` message text, shared by every construction
#: site (including the REPL, which cannot go through :func:`cyclic_value_raise`
#: because it reports a message string, not an exception value).
CYCLE_MESSAGE = "value contains a reference cycle"

#: The single placeholder substituted for a cyclic value by the two call
#: sites that must degrade instead of raising (trace logging, in-flight error
#: reporting) — see ``runtime/trace.py`` and ``pipeline.py``.
CYCLIC_VALUE_MARKER = "<cyclic value>"


class AglCyclicValue(Exception):
    """Sentinel: a structured-value walk re-entered an active value.

    Raised by :func:`enter_value` when rendering or JSON serialization
    revisits a value it has not yet finished visiting. A caller that can
    reach a cyclic value converts this into a catchable ``CyclicValueError``
    via :func:`cyclic_value_raise`.
    """


def enter_value(value_id: int, active: "set[int] | None") -> "set[int]":
    """Mark *value_id* active; raise :class:`AglCyclicValue` on re-entry.

    Cheaper than a ``@contextmanager`` guard (an inline check plus explicit
    ``try``/``finally`` avoids the generator-based context-manager overhead).
    *active* is allocated lazily: every walker's entry point passes ``None``,
    so an acyclic value with no structured values never allocates a set — one
    is created only the first time this is called. The returned set must be
    threaded into every further recursive call so sibling and nested values
    share the same active path, and the caller must ``active.discard(value_id)``
    in a ``finally`` block once done walking *value_id*'s children.
    """
    if active is None:
        active = set()
    elif value_id in active:
        raise AglCyclicValue()
    active.add(value_id)
    return active


def cyclic_value_raise(*, nominals: BuiltinNominals) -> AglRaise:
    """Build the catchable ``AglRaise(CyclicValueError)`` for a detected cycle.

    Single shared constructor so every caller that converts an
    :class:`AglCyclicValue` sentinel produces byte-identical exception
    fields. *nominals* is forwarded to :func:`make_builtin_exception`.
    """
    return AglRaise(
        make_builtin_exception(
            "CyclicValueError",
            CYCLE_MESSAGE,
            nominals=nominals,
        )
    )
