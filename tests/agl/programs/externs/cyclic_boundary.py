def count(xs: list[object]) -> int:
    """Count the outer elements of `xs` without rendering them."""
    return len(xs)


def show(xs: list[object]) -> int:
    """Render the view before counting its outer elements."""
    repr(xs)
    return len(xs)


def identity(x: object) -> object:
    """Return `x` unchanged, but repr it first (forces the sealed handle to
    render its wrapped AgL value, which detects a reference cycle)."""
    repr(x)
    return x
