"""Shared helpers for AgL test modules.

Provides a recursive ``node_id`` collector used by the seeded parsing and
seeded type-checking tests, plus ``ambient_agents_for`` — used by non-scope
unit tests (typecheck/eval/codec/trace) to resolve programs that *call* named
agents without forcing an explicit ``agent`` declaration in every test source.
The agent-declaration RULE itself is exercised by ``tests/test_agl_scope.py``
and the e2e suite; these other modules only need the calls to bind.
"""

from __future__ import annotations

import dataclasses

from agm.agl.syntax.nodes import AgentCall, Program
from agm.agl.syntax.visitor import walk


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


def ambient_agents_for(program: Program) -> frozenset[str]:
    """Return the named-agent call targets in *program* as an ambient set.

    Walks the AST for every ``AgentCall`` and collects each ``agent`` name that
    is a named agent (i.e. not the ``ask``/``exec`` contextual keywords).
    The result is suitable to pass as ``resolve(..., ambient_agents=...)`` so a
    non-scope unit test can resolve a program that calls named agents without
    adding an explicit ``agent`` declaration to its source.
    """
    names: set[str] = set()

    def collect(node: object) -> None:
        if isinstance(node, AgentCall) and node.agent not in ("ask", "exec"):
            names.add(node.agent)

    walk(program, collect)
    return frozenset(names)
