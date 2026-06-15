"""Shared helpers for AgL test modules.

Currently provides a single recursive ``node_id`` collector used by the seeded
parsing and seeded type-checking tests, which both need to know the highest id
consumed by one entry so the next entry's ``start_id`` is disjoint.
"""

from __future__ import annotations

import dataclasses


def all_node_ids(obj: object, seen: set[int] | None = None) -> set[int]:
    """Recursively collect every ``node_id`` reachable from *obj*."""
    if seen is None:
        seen = set()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        nid = getattr(obj, "node_id", None)
        if isinstance(nid, int):
            seen.add(nid)
        for f in dataclasses.fields(obj):
            all_node_ids(getattr(obj, f.name), seen)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            all_node_ids(item, seen)
    return seen
