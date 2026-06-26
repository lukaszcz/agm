"""Unit tests for agm.util.graph — sccs and toposort."""

from __future__ import annotations

import pytest

from agm.util.graph import GraphCycleError, sccs, toposort

# ---------------------------------------------------------------------------
# GraphCycleError
# ---------------------------------------------------------------------------


class TestGraphCycleError:
    def test_stores_cycle_set(self) -> None:
        cycle: set[object] = {"a", "b"}
        exc = GraphCycleError(cycle)
        assert exc.cycle == {"a", "b"}

    def test_str_contains_count(self) -> None:
        exc = GraphCycleError({"x", "y"})
        assert "2" in str(exc)


# ---------------------------------------------------------------------------
# sccs
# ---------------------------------------------------------------------------


class TestSccs:
    def test_empty_graph(self) -> None:
        result = sccs({}, key=str)
        assert result == ()

    def test_single_node_no_edges(self) -> None:
        result = sccs({"a": []}, key=str)
        assert result == (("a",),)

    def test_single_node_self_loop(self) -> None:
        # Self-loop is a trivial SCC of one node.
        result = sccs({"a": ["a"]}, key=str)
        assert result == (("a",),)

    def test_linear_chain_each_own_scc(self) -> None:
        # a -> b -> c: three separate SCCs, reverse-topo order = c, b, a
        adj = {"a": ["b"], "b": ["c"], "c": []}
        result = sccs(adj, key=str)
        # Reverse topo: c (sink) first, then b, then a (root)
        assert result == (("c",), ("b",), ("a",))

    def test_two_node_cycle(self) -> None:
        # a <-> b: one SCC containing both
        adj = {"a": ["b"], "b": ["a"]}
        result = sccs(adj, key=str)
        assert len(result) == 1
        assert set(result[0]) == {"a", "b"}
        # Members sorted by key (str): a before b
        assert result[0] == ("a", "b")

    def test_multi_node_cycle_members_sorted(self) -> None:
        # c -> b -> a -> c: one SCC with members sorted alphabetically
        adj = {"c": ["b"], "b": ["a"], "a": ["c"]}
        result = sccs(adj, key=str)
        assert len(result) == 1
        assert result[0] == ("a", "b", "c")

    def test_two_separate_components(self) -> None:
        # Component 1: a -> b (two singleton SCCs)
        # Component 2: c <-> d (one SCC of two)
        adj = {"a": ["b"], "b": [], "c": ["d"], "d": ["c"]}
        result = sccs(adj, key=str)
        # The 2-node SCC {c,d} has no outgoing edges outside itself: it's a sink SCC.
        # a -> b: b is sink, a depends on b.
        # Expected reverse-topo: ({c,d}, b, a) or ({c,d}, b, a)
        # All nodes form separate SCCs except c,d.
        scc_sets = [set(s) for s in result]
        assert {"c", "d"} in scc_sets
        assert {"b"} in scc_sets
        assert {"a"} in scc_sets
        # c,d SCC must come before a (no ordering constraint between c,d and a/b)
        # b must come before a (a depends on b).
        b_idx = next(i for i, s in enumerate(result) if s == ("b",))
        a_idx = next(i for i, s in enumerate(result) if s == ("a",))
        assert b_idx < a_idx

    def test_reverse_topological_ordering_explicit(self) -> None:
        # root -> middle -> leaf: SCCs should be (leaf, middle, root)
        adj = {"root": ["middle"], "middle": ["leaf"], "leaf": []}
        result = sccs(adj, key=str)
        assert result == (("leaf",), ("middle",), ("root",))

    def test_within_scc_sort_by_key(self) -> None:
        # Three nodes in a cycle; key is reversed string to test custom ordering.
        adj = {"apple": ["banana"], "banana": ["cherry"], "cherry": ["apple"]}
        result = sccs(adj, key=lambda s: s[::-1])
        # All three in one SCC, sorted by reversed string:
        # "apple" -> "elppa", "banana" -> "ananab", "cherry" -> "yrrehc"
        # alphabetically: "ananab" < "elppa" < "yrrehc"
        assert len(result) == 1
        assert result[0] == ("banana", "apple", "cherry")

    def test_node_not_in_adj_as_successor(self) -> None:
        # b is a successor of a but has no entry in adj
        adj = {"a": ["b"]}
        result = sccs(adj, key=str)
        # b gets discovered during DFS but is not in adj keys
        # b has no outgoing edges -> singleton SCC
        scc_sets = [set(s) for s in result]
        assert {"b"} in scc_sets
        assert {"a"} in scc_sets
        b_idx = next(i for i, s in enumerate(result) if "b" in s)
        a_idx = next(i for i, s in enumerate(result) if "a" in s)
        assert b_idx < a_idx

    def test_integer_nodes(self) -> None:
        # Test with integer nodes to verify generic typing works.
        adj = {1: [2], 2: [3], 3: []}
        result = sccs(adj, key=lambda x: x)
        assert result == ((3,), (2,), (1,))


# ---------------------------------------------------------------------------
# toposort
# ---------------------------------------------------------------------------


class TestToposort:
    def test_empty_nodes(self) -> None:
        result = toposort([], {}, key=str)
        assert result == []

    def test_single_node_no_deps(self) -> None:
        result = toposort(["a"], {}, key=str)
        assert result == ["a"]

    def test_linear_chain(self) -> None:
        # a depends on b depends on c: order should be c, b, a
        nodes = ["a", "b", "c"]
        deps = {"a": ["b"], "b": ["c"], "c": []}
        result = toposort(nodes, deps, key=str)
        assert result.index("c") < result.index("b") < result.index("a")

    def test_diamond_dependency(self) -> None:
        # d depends on b and c, both depend on a
        nodes = ["a", "b", "c", "d"]
        deps = {"b": ["a"], "c": ["a"], "d": ["b", "c"]}
        result = toposort(nodes, deps, key=str)
        assert result.index("a") < result.index("b")
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")

    def test_deterministic_tie_breaking(self) -> None:
        # a and b both have no dependencies: key determines order.
        nodes = ["b", "a"]
        deps: dict[str, list[str]] = {}
        result = toposort(nodes, deps, key=str)
        # str key: "a" < "b", so a should come first.
        assert result == ["a", "b"]

    def test_two_node_cycle_raises(self) -> None:
        nodes = ["a", "b"]
        deps = {"a": ["b"], "b": ["a"]}
        with pytest.raises(GraphCycleError) as exc_info:
            toposort(nodes, deps, key=str)
        assert exc_info.value.cycle == {"a", "b"}

    def test_longer_cycle_raises(self) -> None:
        nodes = ["a", "b", "c"]
        deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(GraphCycleError) as exc_info:
            toposort(nodes, deps, key=str)
        assert exc_info.value.cycle == {"a", "b", "c"}

    def test_partial_cycle_independent_node_not_in_cycle(self) -> None:
        # a has no dependencies; b and c form a cycle.
        # a is orderable; b and c are not.
        nodes = ["a", "b", "c"]
        deps = {"b": ["c"], "c": ["b"]}
        with pytest.raises(GraphCycleError) as exc_info:
            toposort(nodes, deps, key=str)
        # Only b and c are unorderable; a was successfully placed.
        assert exc_info.value.cycle == {"b", "c"}

    def test_partial_cycle_dependent_also_unorderable(self) -> None:
        # a depends on b which is in a cycle with c; all three are unorderable.
        nodes = ["a", "b", "c"]
        deps = {"a": ["b"], "b": ["c"], "c": ["b"]}
        with pytest.raises(GraphCycleError) as exc_info:
            toposort(nodes, deps, key=str)
        # a can never be ordered because b is stuck in a cycle.
        assert exc_info.value.cycle == {"a", "b", "c"}

    def test_node_absent_from_deps(self) -> None:
        # nodes present but not in deps dict: treated as having no dependencies.
        nodes = ["a", "b", "c"]
        deps = {"c": ["a"]}
        result = toposort(nodes, deps, key=str)
        # a and b have no deps (b absent from deps entirely).
        # c depends on a.
        assert result.index("a") < result.index("c")
        # b should appear (before c, since b has no deps).
        assert "b" in result

    def test_reverse_key_ordering(self) -> None:
        # Nodes with no deps, reverse-alphabet key.
        nodes = ["a", "b", "c"]
        deps: dict[str, list[str]] = {}
        result = toposort(nodes, deps, key=lambda s: [-ord(c) for c in s])
        # Reverse alphabetical: c, b, a
        assert result == ["c", "b", "a"]

    def test_integer_nodes(self) -> None:
        nodes = [3, 1, 2]
        deps = {3: [1], 2: [1]}
        result = toposort(nodes, deps, key=lambda x: x)
        assert result[0] == 1
        # 2 and 3 both depend on 1; 2 < 3 by key.
        assert result[1] == 2
        assert result[2] == 3
