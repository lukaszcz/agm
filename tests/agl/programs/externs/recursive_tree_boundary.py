from agl import Tree


def build_tree(n: int) -> Tree:
    if n <= 0:
        return Tree.Leaf(value=1)
    return Tree.Node(left=Tree.Leaf(value=1), right=build_tree(n - 1))


def sum_tree(tree: Tree) -> int:
    match tree:
        case Tree.Leaf(value=value):
            return value
        case Tree.Node(left=left, right=right):
            return sum_tree(left) + sum_tree(right)
