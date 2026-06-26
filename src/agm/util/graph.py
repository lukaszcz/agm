"""Pure, generic graph algorithms for AGM (stdlib-only, zero agm imports).

Provides:
- ``GraphCycleError`` — raised by :func:`toposort` when a cycle is detected.
- ``sccs`` — Tarjan's strongly-connected-components (reverse topological order).
- ``toposort`` — Kahn's topological sort (leaves first, deterministic).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

T = TypeVar("T")
K = TypeVar("K", bound="SupportsRichComparison")

__all__ = ["GraphCycleError", "sccs", "toposort"]


class GraphCycleError(Exception):
    """Raised by :func:`toposort` when the dependency graph contains a cycle.

    ``cycle`` is the set of nodes that could not be ordered (the participants
    in the cycle, expressed as ``set[object]`` so no TypeVar threading through
    the exception class is required).
    """

    def __init__(self, cycle: set[object]) -> None:
        self.cycle = cycle
        super().__init__(f"cycle detected among {len(cycle)} node(s)")


def sccs(
    adj: Mapping[T, Iterable[T]],
    *,
    key: Callable[[T], K],
) -> tuple[tuple[T, ...], ...]:
    """Return the strongly-connected components of *adj* in reverse topological order.

    Uses Tarjan's algorithm.  Each SCC's members are sorted by *key*; nodes are
    visited in *key*-sorted order so the result is fully deterministic.

    Parameters
    ----------
    adj:
        Adjacency mapping: ``adj[node]`` yields the direct successors of *node*.
        Nodes absent from *adj* are treated as having no outgoing edges.
    key:
        Sort key applied to nodes for deterministic ordering within each SCC and
        for the outer iteration order.

    Returns
    -------
    tuple[tuple[T, ...], ...]
        SCCs in **reverse topological order** (sinks first, roots last).  Within
        each SCC the members are sorted by *key*.
    """
    index_counter = [0]
    stack: list[T] = []
    on_stack: set[T] = set()
    index: dict[T, int] = {}
    lowlink: dict[T, int] = {}
    result: list[tuple[T, ...]] = []

    def strongconnect(node: T) -> None:
        idx = index_counter[0]
        index[node] = idx
        lowlink[node] = idx
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        if node in adj:
            for neighbour in adj[node]:
                if neighbour not in index:
                    strongconnect(neighbour)
                    lowlink[node] = min(lowlink[node], lowlink[neighbour])
                elif neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], index[neighbour])

        if lowlink[node] == index[node]:
            scc: list[T] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == node:
                    break
            scc.sort(key=key)
            result.append(tuple(scc))

    nodes = sorted(adj, key=key)
    for node in nodes:
        if node not in index:
            strongconnect(node)

    return tuple(result)


def toposort(
    nodes: Iterable[T],
    deps: Mapping[T, Iterable[T]],
    *,
    key: Callable[[T], K],
) -> list[T]:
    """Return *nodes* in topological order (leaves first) via Kahn's algorithm.

    ``deps[n]`` is the collection of nodes that *n* **depends on** (must appear
    before *n* in the output).  Ties between ready nodes are broken by *key* for
    determinism.

    Parameters
    ----------
    nodes:
        All nodes to be sorted.  Must be unique (no duplicates); duplicates
        would make the cycle check (ordered-count vs node-count) spurious.
    deps:
        Dependency mapping: ``deps[n]`` yields the nodes that *n* depends on.
        Nodes absent from *deps* are treated as having no dependencies.
    key:
        Sort key for deterministic tie-breaking in the ready queue.

    Returns
    -------
    list[T]
        All *nodes* ordered so each node appears after all its dependencies.

    Raises
    ------
    GraphCycleError
        When the graph contains a cycle.  ``exc.cycle`` is the set of nodes
        that could not be placed (the cycle participants).
    """
    all_nodes = list(nodes)

    # Build in-degree and reverse-adjacency (adj[u] = nodes that depend on u).
    in_degree: dict[T, int] = {k: 0 for k in all_nodes}
    adj: dict[T, list[T]] = {k: [] for k in all_nodes}

    for node, node_deps in deps.items():
        for dep in node_deps:
            adj[dep].append(node)
            in_degree[node] = in_degree.get(node, 0) + 1

    # Kahn's: start with all zero-in-degree nodes, sorted for determinism.
    ready: list[T] = sorted(
        (k for k, d in in_degree.items() if d == 0),
        key=key,
    )
    order: list[T] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        dependents = sorted(adj.get(node, []), key=key)
        for dep in dependents:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                bisect.insort(ready, dep, key=key)

    if len(order) < len(all_nodes):
        done = set(order)
        remaining: set[object] = set()
        for k in all_nodes:
            if k not in done:
                remaining.add(k)
        raise GraphCycleError(remaining)

    return order
