def build_tree(n: int) -> dict[str, object]:
    """Build a right-leaning Tree of `n` internal Nodes, each left-branch a Leaf(1).

    Returned as the nested ``{"$case": ..., ...}`` dict shape the boundary
    decodes; a value of 1 per leaf makes the total leaf count easy to predict.
    """
    if n <= 0:
        return {"$case": "Leaf", "value": 1}
    return {
        "$case": "Node",
        "left": {"$case": "Leaf", "value": 1},
        "right": build_tree(n - 1),
    }


def sum_tree(t: dict[str, object]) -> int:
    """Sum every Leaf value in the boundary-encoded Tree dict `t`."""
    if t["$case"] == "Leaf":
        value = t["value"]
        assert isinstance(value, int)
        return value
    left = t["left"]
    right = t["right"]
    assert isinstance(left, dict) and isinstance(right, dict)
    return sum_tree(left) + sum_tree(right)
