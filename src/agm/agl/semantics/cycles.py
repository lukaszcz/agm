"""Cycle detection shared by the AgL container walkers.

Reference semantics makes cyclic ``array``/``dict`` values constructible: an
array or dict can hold a reference back to a container that (transitively)
contains it. Every walker that recurses through a value's containers —
rendering, JSON serialization, and the Python FFI encoder — must detect that
re-entry rather than recursing forever. A cycle can only ever be closed
through an array or a dict: records, enums, and exceptions are immutable, so
none of them can hold a reference to itself. Tracking container identity is
therefore enough; no other value kind ever needs to join the active set.

This module is the single shared implementation of that active-set walk, so
``render_value``, ``value_to_json_obj``, and the FFI boundary encoder do not
each reimplement it. Equality is unrelated — it is co-inductive
(``semantics/values.py``) rather than error-raising, and does not use this
module.
"""

from __future__ import annotations

from agm.agl.semantics.exceptions import AglRaise, make_builtin_exception

__all__ = [
    "CYCLE_MESSAGE",
    "CYCLIC_VALUE_MARKER",
    "AglCyclicValue",
    "cyclic_value_raise",
    "enter_container",
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
    """Sentinel: a container walk re-entered a container already on its own path.

    Raised by :func:`enter_container` when a walker (rendering, JSON
    serialization, the FFI encoder) revisits a container it has not yet
    finished visiting. A caller that can reach a cyclic value converts this
    into a catchable ``CyclicValueError`` via :func:`cyclic_value_raise`.
    """


def enter_container(container_id: int, active: "set[int] | None") -> "set[int]":
    """Mark *container_id* active; raise :class:`AglCyclicValue` on re-entry.

    Cheaper than a ``@contextmanager`` guard (an inline check plus explicit
    ``try``/``finally`` avoids the generator-based context-manager overhead).
    *active* is allocated
    lazily: every walker's entry point passes ``None``, so an acyclic value
    with no containers never allocates a set — one is created only the first
    time this is called. The returned set must be threaded into every further
    recursive call so sibling and nested containers share the same active
    path, and the caller must ``active.discard(container_id)`` in a
    ``finally`` block once done walking *container_id*'s contents.
    """
    if active is None:
        active = set()
    elif container_id in active:
        raise AglCyclicValue()
    active.add(container_id)
    return active


def cyclic_value_raise(trace_id: str) -> AglRaise:
    """Build the catchable ``AglRaise(CyclicValueError)`` for a detected cycle.

    Single shared constructor so every caller that converts an
    :class:`AglCyclicValue` sentinel produces byte-identical exception
    fields.
    """
    return AglRaise(
        make_builtin_exception(
            "CyclicValueError",
            CYCLE_MESSAGE,
            trace_id=trace_id,
        )
    )
